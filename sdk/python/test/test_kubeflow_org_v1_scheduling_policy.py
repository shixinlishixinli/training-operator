# coding: utf-8

"""
    Kubeflow Training SDK

    Python SDK for Kubeflow Training  # noqa: E501

    The version of the OpenAPI document: v1.7.0
    Generated by: https://openapi-generator.tech
"""


from __future__ import absolute_import

import unittest
import datetime

from kubeflow.training.models import *
from kubeflow.training.models.kubeflow_org_v1_scheduling_policy import KubeflowOrgV1SchedulingPolicy  # noqa: E501
from kubeflow.training.rest import ApiException

class TestKubeflowOrgV1SchedulingPolicy(unittest.TestCase):
    """KubeflowOrgV1SchedulingPolicy unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional):
        """Test KubeflowOrgV1SchedulingPolicy
            include_option is a boolean, when False only required
            params are included, when True both required and
            optional params are included """
        # model = kubeflow.training.models.kubeflow_org_v1_scheduling_policy.KubeflowOrgV1SchedulingPolicy()  # noqa: E501
        if include_optional :
            return KubeflowOrgV1SchedulingPolicy(
                min_available = 56, 
                min_resources = {
                    'key' : None
                    }, 
                priority_class = '0', 
                queue = '0', 
                schedule_timeout_seconds = 56
            )
        else :
            return KubeflowOrgV1SchedulingPolicy(
        )

    def testKubeflowOrgV1SchedulingPolicy(self):
        """Test KubeflowOrgV1SchedulingPolicy"""
        inst_req_only = self.make_instance(include_optional=False)
        inst_req_and_optional = self.make_instance(include_optional=True)


if __name__ == '__main__':
    unittest.main()
