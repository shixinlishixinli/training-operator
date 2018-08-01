package controller

import (
	"fmt"
	"time"

	log "github.com/sirupsen/logrus"
	"k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1unstructured "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/kubernetes/scheme"

	tfv1alpha2 "github.com/kubeflow/tf-operator/pkg/apis/tensorflow/v1alpha2"
)

const (
	failedMarshalTFJobReason = "FailedMarshalTFJob"
	terminatedTFJobReason    = "TFJobTerminated"
)

// When a pod is added, set the defaults and enqueue the current tfjob.
func (tc *TFJobController) addTFJob(obj interface{}) {
	// Convert from unstructured object.
	tfJob, err := tfJobFromUnstructured(obj)
	if err != nil {
		un, ok := obj.(*metav1unstructured.Unstructured)
		logger := &log.Entry{}
		if ok {
			logger = loggerForUnstructured(un)
		}
		logger.Errorf("Failed to convert the TFJob: %v", err)
		// Log the failure to conditions.
		if err == errFailedMarshal {
			errMsg := fmt.Sprintf("Failed to unmarshal the object to TFJob object: %v", err)
			logger.Warn(errMsg)
			tc.recorder.Event(tfJob, v1.EventTypeWarning, failedMarshalTFJobReason, errMsg)
		}
		return
	}

	// Set default for the new tfjob.
	scheme.Scheme.Default(tfJob)

	msg := fmt.Sprintf("TFJob %s is created.", tfJob.Name)
	logger := loggerForTFJob(tfJob)
	logger.Info(msg)

	// Add a created condition.
	err = updateTFJobConditions(tfJob, tfv1alpha2.TFJobCreated, tfJobCreatedReason, msg)
	if err != nil {
		logger.Errorf("Append tfJob condition error: %v", err)
		return
	}

	// Convert from tfjob object
	err = unstructuredFromTFJob(obj, tfJob)
	if err != nil {
		logger.Error("Failed to convert the obj: %v", err)
		return
	}
	tc.enqueueTFJob(obj)
}

// When a pod is updated, enqueue the current tfjob.
func (tc *TFJobController) updateTFJob(old, cur interface{}) {
	oldTFJob, err := tfJobFromUnstructured(old)
	if err != nil {
		return
	}
	log.Infof("Updating tfjob: %s", oldTFJob.Name)
	tc.enqueueTFJob(cur)
}

func (tc *TFJobController) deletePdb(tfJob *tfv1alpha2.TFJob) error {

	// Check the pdb exist or not
	_, err := tc.kubeClientSet.PolicyV1beta1().PodDisruptionBudgets(tfJob.Namespace).Get(tfJob.Name, metav1.GetOptions{})
	if err != nil && k8serrors.IsNotFound(err) {
		return nil
	}

	tc.recorder.Event(tfJob, v1.EventTypeNormal, terminatedTFJobReason,
		"TFJob is terminated, deleting pdb")

	msg := fmt.Sprintf("Deleting pdb %s", tfJob.Name)
	log.Info(msg)

	if err := tc.kubeClientSet.PolicyV1beta1().PodDisruptionBudgets(tfJob.Namespace).Delete(tfJob.Name, &metav1.DeleteOptions{}); err != nil {
		tc.recorder.Eventf(tfJob, v1.EventTypeWarning, "FailedDeletePdb", "Error deleting: %v", err)
		return fmt.Errorf("unable to delete pdb: %v", err)
	} else {
		tc.recorder.Eventf(tfJob, v1.EventTypeNormal, "SuccessfulDeletePdb", "Deleted pdb: %v", tfJob.Name)
	}

	return nil
}

func (tc *TFJobController) deletePodsAndServices(tfJob *tfv1alpha2.TFJob, pods []*v1.Pod) error {
	if len(pods) == 0 {
		return nil
	}
	tc.recorder.Event(tfJob, v1.EventTypeNormal, terminatedTFJobReason,
		"TFJob is terminated, deleting pods and services")

	// Delete nothing when the cleanPodPolicy is None.
	if *tfJob.Spec.CleanPodPolicy == tfv1alpha2.CleanPodPolicyNone {
		return nil
	}

	for _, pod := range pods {
		if *tfJob.Spec.CleanPodPolicy == tfv1alpha2.CleanPodPolicyRunning && pod.Status.Phase != v1.PodRunning {
			continue
		}
		if err := tc.podControl.DeletePod(pod.Namespace, pod.Name, tfJob); err != nil {
			return err
		}
		// Pod and service have the same name, thus the service could be deleted using pod's name.
		if err := tc.serviceControl.DeleteService(pod.Namespace, pod.Name, tfJob); err != nil {
			return err
		}
	}
	return nil
}

func (tc *TFJobController) cleanupTFJob(tfJob *tfv1alpha2.TFJob) error {
	currentTime := time.Now()
	ttl := tfJob.Spec.TTLSecondsAfterFinished
	if ttl == nil {
		// do nothing if the cleanup delay is not set
		return nil
	}
	duration := time.Second * time.Duration(*ttl)
	if currentTime.After(tfJob.Status.CompletionTime.Add(duration)) {
		err := tc.deleteTFJobHandler(tfJob)
		if err != nil {
			loggerForTFJob(tfJob).Warnf("Cleanup TFJob error: %v.", err)
			return err
		}
		return nil
	}
	key, err := KeyFunc(tfJob)
	if err != nil {
		loggerForTFJob(tfJob).Warnf("Couldn't get key for tfjob object: %v", err)
		return err
	}
	tc.workQueue.AddRateLimited(key)
	return nil
}

// deleteTFJob delets the given TFJob.
func (tc *TFJobController) deleteTFJob(tfJob *tfv1alpha2.TFJob) error {
	return tc.tfJobClientSet.KubeflowV1alpha2().TFJobs(tfJob.Namespace).Delete(tfJob.Name, &metav1.DeleteOptions{})
}
