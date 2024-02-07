# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Methods for TensorBoard apps hosted on SageMaker.

This module contains methods for starting up and accessing
TensorBoard apps hosted on SageMaker
"""
from __future__ import absolute_import

import logging

from typing import Optional

from sagemaker.interactive_apps.base_interactive_app import BaseInteractiveApp

logger = logging.getLogger(__name__)


class TensorBoardApp(BaseInteractiveApp):
    """TensorBoardApp is a class for creating/accessing a TensorBoard app hosted on Studio."""

    def get_app_url(
        self,
        training_job_name: Optional[str] = None,
        open_in_default_web_browser: Optional[bool] = True,
        create_presigned_domain_url: Optional[bool] = False,
        domain_id: Optional[str] = None,
        user_profile_name: Optional[str] = None,
        optional_create_presigned_url_kwargs: Optional[dict] = None,
    ):
        """Generate a URL to help access the TensorBoard application hosted in Studio.

           For users that are already in SageMaker Studio, this method tries to get the
           domain id and the user profile from the Studio environment. If successful, the generated
           URL will direct to the TensorBoard application in SageMaker. Otherwise, it will direct
           to the TensorBoard landing page in the SageMaker console. If a user outside of SageMaker
           Studio passes in a valid domain ID and user profile name, the generated URL will be
           presigned - authenticating the user and redirecting to the TensorBoard app once used.
           Otherwise, the URL will direct to the TensorBoard landing page in the SageMaker console.
           By default, the generated URL will attempt to open in the environment's default web
           browser.

        Args:
            training_job_name (str): Optional. The name of the training job to pre-load in
                TensorBoard. If nothing provided, the method just returns the TensorBoard
                application URL. You can add training jobs later by using the SageMaker Data
                Manager UI.
                Default: ``None``
            open_in_default_web_browser (bool): Optional. When True, the URL will attempt to be
                opened in the environment's default web browser. Otherwise, the resulting URL will
                be returned by this function.
                Default: ``True``
            create_presigned_domain_url (bool): Optional. Determines whether a presigned domain URL
                should be generated instead of an unsigned URL. This only applies when called from
                outside of a SageMaker Studio environment. If this is set to True inside of a
                SageMaker Studio environment, it will be ignored.
                Default: ``False``
            domain_id (str): Optional. This parameter should be passed when a user outside of
                Studio wants a presigned URL to the TensorBoard application. This value will map to
                'DomainId' in the resulting create_presigned_domain_url call. Must be passed with
                user_profile_name and create_presigned_domain_url set to True.
                Default: ``None``
            user_profile_name (str): Optional. This parameter should be passed when a user outside
                of Studio wants a presigned URL to the TensorBoard application. This value will
                map to 'UserProfileName' in the resulting create_presigned_domain_url call. Must be
                passed with domain_id and create_presigned_domain_url set to True.
                Default: ``None``
            optional_create_presigned_url_kwargs (dict): Optional. This parameter
                should be passed when a user outside of Studio wants a presigned URL to the
                TensorBoard application and wants to modify the optional parameters of the
                create_presigned_domain_url call.
                Default: ``None``

        Returns:
            str: A URL for TensorBoard hosted on SageMaker.
        """
        if training_job_name is not None:
            self._validate_job_name(training_job_name)

        if optional_create_presigned_url_kwargs is None:
            optional_create_presigned_url_kwargs = {}

        if domain_id is not None:
            optional_create_presigned_url_kwargs["DomainId"] = domain_id

        if user_profile_name is not None:
            optional_create_presigned_url_kwargs["UserProfileName"] = user_profile_name

        if (
            create_presigned_domain_url
            and not self._is_in_studio()
            and self._validate_domain_id(optional_create_presigned_url_kwargs.get("DomainId"))
            and self._validate_user_profile_name(
                optional_create_presigned_url_kwargs.get("UserProfileName")
            )
        ):
            state_to_encode = None
            redirect = "TensorBoard"

            if training_job_name is not None:
                state_to_encode = (
                    "/tensorboard/default/data/plugin/sagemaker_data_manager/"
                    + f"add_folder_or_job?Redirect=True&Name={training_job_name}"
                )

            url = self._get_presigned_url(
                optional_create_presigned_url_kwargs, redirect, state_to_encode
            )
        elif self._is_in_studio() and self._validate_domain_and_user():
            if domain_id or user_profile_name:
                logger.warning(
                    "Ignoring passed in domain_id and user_profile_name for Studio set values."
                )
            url = (
                f"https://{self._domain_id}.studio.{self.region}."
                + "sagemaker.aws/tensorboard/default"
            )
            if training_job_name is not None:
                self._validate_job_name(training_job_name)
                url += (
                    "/data/plugin/sagemaker_data_manager/"
                    + f"add_folder_or_job?Redirect=True&Name={training_job_name}"
                )
            else:
                url += "/#sagemaker_data_manager"
        else:
            if domain_id or user_profile_name or create_presigned_domain_url:
                logger.warning(
                    "A valid domain ID and user profile name were not provided. "
                    "Providing default landing page URL as a result."
                )
            url = (
                f"https://{self.region}.console.aws.amazon.com/sagemaker/home"
                + f"?region={self.region}#/tensor-board-landing"
            )
            if training_job_name is not None:
                url += f"/{training_job_name}"

        if open_in_default_web_browser:
            self._open_url_in_web_browser(url)
            url = ""
        return url
