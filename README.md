
# Projects and Labs in Data Science using Machine Learning Pipelines 

##  ML_Pipelines_using_BERT

### Build, Train, and Deploy ML Pipelines using BERT

### Summary of my Labs and Projects from Coursera

- Automate a natural language processing task by building an end-to-end machine learning pipeline using Hugging Face’s highly-optimized implementation of the state-of-the-art BERT algorithm with Amazon SageMaker Pipelines. The pipeline will first transform the dataset into BERT-readable features and store the features in the Amazon SageMaker Feature Store. It will then fine-tune a text classification model to the dataset using a Hugging Face pre-trained model, which has learned to understand the human language from millions of Wikipedia documents. Finally, your pipeline will evaluate the model’s accuracy and only deploy the model if the accuracy exceeds a given threshold.

- Practical data science is geared towards handling massive datasets that do not fit in your local hardware and could originate from multiple sources. One of the biggest benefits of developing and running data science projects in the cloud is the agility and elasticity that the cloud offers to scale up and out at a minimum cost.

- The focus of these projects are on developing ML workflows using Amazon SageMaker, Python and SQL programming languages. I fully recommend you these projects if you want to learn how to build, train, and deploy scalable, end-to-end ML pipelines in the AWS cloud.

- Steps ML_Pipelines_using_BERT

    1. Configure the SageMaker Feature Store

    2. Transform the dataset

    3. Inspect the transformed data

    4. Query the Feature Store

- Train a review classifier with BERT and Amazon SageMaker

    5. Configure dataset

    6. Configure model hyper-parameters

    7. Setup evaluation metrics, debugger and profiler

    8. Train model

    9. Analyze debugger results

    10. Deploy and test the model
    
- SageMaker pipelines to train a BERT-Based text classifier

    11. Configure dataset and processing step

    12. Configure training step

    13. Configure model-evaluation step

    14. Configure register model step

    15. Create model for deployment step

    16. Check accuracy condition step

    17. Create and start pipeline

    18. List pipeline artifacts

    19. Approve and deploy model
    
    
# SageMaker pipelines to train a BERT-Based text classifier

### My Quick Lab from Practical Data Science Specializastion, Coursera.

In this lab, you will do the following:
* Define and run a pipeline using a directed acyclic graph (DAG) with specific pipeline parameters and model hyper-parameters
* Define a processing step that cleans, balances, transforms, and splits our dataset into train, validation, and test dataset
* Define a training step that trains a model using the train and validation datasets
* Define a processing step that evaluates the trained model's performance on the test dataset
* Define a register model step that creates a model package from the trained model
* Define a conditional step that checks the model's performance and conditionally registers the model for deployment

**Terminology**

This notebook focuses on the following features of Amazon SageMaker Pipelines:

* **Pipelines** - a directed acyclic graph (DAG) of steps and conditions to orchestrate SageMaker jobs and resource creation
* **Processing job steps** - a simplified, managed experience on SageMaker to run data processing workloads, such as feature engineering, data validation, model evaluation, and model explainability
* **Training job steps** - an iterative process that teaches a model to make predictions on new data by presenting examples from a training dataset
* **Conditional step execution** - provides conditional execution of branches in a pipeline
* **Registering models** - register a model in a model registry to create a deployable models in Amazon SageMaker
* **Parameterized pipeline executions** - allows pipeline executions to vary by supplied parameters
* **Model endpoint** - hosts the model as a REST endpoint to serve predictions from new data


1. Configure dataset and processing step

2. Configure training step

3. Configure model-evaluation step

4. Configure register model step

5. Create model for deployment step

6. Check accuracy condition step

7. Create and start pipeline

8. List pipeline artifacts

9. Approve and deploy model

**BERT Pipeline**

The pipeline that you will create follows a typical machine learning application pattern of pre-processing, training, evaluation, and model registration.

In the processing step, you will perform feature engineering to transform the `review_body` text into BERT embeddings using the pre-trained BERT model and split the dataset into train, validation and test files. The transformed dataset is stored in a feature store. To optimize for Tensorflow training, the transformed dataset files are saved using the TFRecord format in Amazon S3.

In the training step, you will fine-tune the BERT model to the customer reviews dataset and add a new classification layer to predict the `sentiment` for a given `review_body`.

In the evaluation step, you will take the trained model and a test dataset as input, and produce a JSON file containing classification evaluation metrics.

In the condition step, you will register the trained model if the accuracy of the model, as determined by our evaluation step, exceeds a given threshold value. 

![](./images/bert_sagemaker_pipeline.png)


    
References:

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805v2.pdf)
    
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

- [Fundamental Techniques of Feature Engineering for Machine Learning](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)

- [PyTorch Hub](https://pytorch.org/hub/)

- [TensorFlow Hub](https://www.tensorflow.org/hub)

- [Hugging Face open-source NLP transformers library](https://github.com/huggingface/transformers) 

- [RoBERTa model](https://arxiv.org/abs/1907.11692) 

- [Amazon SageMaker Model Training (Developer Guide)](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html)

- [Amazon SageMaker Debugger: A system for real-time insights into machine learning model training](https://www.amazon.science/publications/amazon-sagemaker-debugger-a-system-for-real-time-insights-into-machine-learning-model-training)

- [The science behind SageMaker’s cost-saving Debugger](https://www.amazoN.science/blog/the-science-behind-sagemakers-cost-saving-debugger)

- [Amazon SageMaker Debugger (Developer Guide)](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html)

- [Amazon SageMaker Debugger (GitHub)](https://github.com/awslabs/sagemaker-debugger)



# Analyze Datasets and Train ML Models using AutoML

### Register and visualize dataset

    List and access the Women's Clothing Reviews dataset files hosted in an S3 bucket

    Install and import AWS Data Wrangler

    Create an AWS Glue Catalog database and list all Glue Catalog databases

    Register dataset files with the AWS Glue Catalog

    Write SQL queries to answer specific questions on your dataset and run your queries with Amazon Athena

    Return the query results in a pandas dataframe

    Produce and select different plots and visualizations that address your questions






