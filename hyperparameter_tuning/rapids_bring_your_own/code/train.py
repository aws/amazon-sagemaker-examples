#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys, os, time, traceback
from rapids_cloud_ml import RapidsCloudML 

""" Airline dataset specific target variable and feature column names 
    Note: If you plan to bring in your own dataset modify the variables below
""" 
dataset_label_column = 'ArrDel15'
dataset_feature_columns = [ 'Year', 'Quarter', 'Month', 'DayOfWeek', 
                            'Flight_Number_Reporting_Airline', 'DOT_ID_Reporting_Airline',
                            'OriginCityMarketID', 'DestCityMarketID',
                            'DepTime', 'DepDelay', 'DepDel15', 'ArrDel15',
                            'AirTime', 'Distance' ]

if __name__ == "__main__":

    start_time = time.time()

    # parse inputs and build cluster
    rapids_sagemaker = RapidsCloudML ( input_args = sys.argv[1:] )
    
    try:
        print( '--- starting workflow --- \n ')

        # [ optional cross-validation] improves robustness/confidence in the best hyper-params        
        for i_fold in range ( rapids_sagemaker.cv_folds ):

            # run ETL [  ingest -> repartition -> drop missing -> split -> persist ]
            X_train, X_test, y_train, y_test = rapids_sagemaker.ETL ( columns = dataset_feature_columns, 
                                                                      label_column = dataset_label_column,
                                                                      random_seed = i_fold ) 

            # train model
            trained_model = rapids_sagemaker.train_model ( X_train, y_train )

            # evaluate perf
            score = rapids_sagemaker.predict ( trained_model, X_test, y_test )

            # restart cluster to avoid memory creep [ for multi-CPU/GPU ]
            rapids_sagemaker.cluster_reinitialize( i_fold )

        # save
        rapids_sagemaker.save_model ( trained_model )
                
        # emit final score to sagemaker
        rapids_sagemaker.emit_final_score()
                        
        print( f'total elapsed time = { round( time.time() - start_time) } seconds\n' )
    
        sys.exit(0) # success exit code

    except Exception as error:

        trc = traceback.format_exc()           
        print( ' ! exception: ' + str(error) + '\n' + trc, file = sys.stderr)
        
        sys.exit(-1) # a non-zero exit code causes the training job to be marked as failed