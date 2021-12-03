
#Understanding Company Valuation through Sentiment Analysis of Earnings and News 

## End-to-end NLP MLOps: Orchestrating company earnings trend analysis, using SEC filings and news sentiment with HuggingFace Transformers, using Amazon Sagemaker Pipelines

We are going to demonstrate how to summarize and derive sentiments out of Security and Exchange Commission reports filed by a publicly traded organization. We are also going to derive the overall market sentiments about the said organization through financial news articles within the same financial period to present a fair view of the organization vs. market sentiments and outlook about the company's overall valuation and performance. In addition to this we will also identify the most popular keywords and entities within the news articles about that organization.

In order to achieve the above we will be using multiple SageMaker Huggingface based NLP transformers for the downstream NLP tasks of Summarization (e.g., of the news and SEC MDNA sections) and Sentiment Analysis (of the resulting summaries).

### Using SageMaker Pipelines
Amazon SageMaker Pipelines is the first purpose-built, easy-to-use continuous integration and continuous delivery (CI/CD) service for machine learning (ML). With SageMaker Pipelines, you can create, automate, and manage end-to-end ML workflows at scale. 

Orchestrating workflows across each step of the machine learning process (e.g. exploring and preparing data, experimenting with different algorithms and parameters, training and tuning models, and deploying models to production) can take months of coding.

Since it is purpose-built for machine learning, SageMaker Pipelines helps you automate different steps of the ML workflow, including data loading, data transformation, training and tuning, and deployment. With SageMaker Pipelines, you can build dozens of ML models a week, manage massive volumes of data, thousands of training experiments, and hundreds of different model versions. You can share and re-use workflows to recreate or optimize models, helping you scale ML throughout your organization.

### Understanding trends in company valuation (or similar) with NLP

**Natural language processing (NLP)** is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves. (Source: [Wikipedia](https://en.wikipedia.org/wiki/Natural_language_processing))

* Summarization of financial text from SEC reports and news articles will be done via [Pegasus for Financial Summarization model](https://huggingface.co/human-centered-summarization/financial-summarization-pegasus) based on the paper [Towards Human-Centered Summarization: A Case Study on Financial News](https://aclanthology.org/2021.hcinlp-1.4/). 
* Sentiment analysis on summarized SEC financial report and news articles will be done via pre-trained NLP model to analyze sentiment of financial text called [FinBERT](https://huggingface.co/ProsusAI/finbert). Paper: [ FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)
* Popular Keyword and named entity recognition using Amazon Comprehend.

### SEC Dataset

The starting point for a vast amount of financial NLP is text in SEC filings. The SEC requires companies to report different types of information related to various events involving companies. The full list of SEC forms is here: https://www.sec.gov/forms.

SEC filings are widely used by financial services companies as a source of information about companies in order to make trading, lending, investment, and risk management decisions. Because these filings are required by regulation, they are of high quality and veracity. They contain forward-looking information that helps with forecasts and are written with a view to the future, required by regulation. In addition, in recent times, the value of historical time-series data has degraded, since economies have been structurally transformed by trade wars, pandemics, and political upheavals. Therefore, text as a source of forward-looking information has been increasing in relevance. 

#### How to obtain the data?

Downloading SEC filings is done from the SEC's Electronic Data Gathering, Analysis, and Retrieval (EDGAR) website, which provides open data access. EDGAR is the primary system under the U.S. Securities And Exchange Commission (SEC) for companies and others submitting documents under the Securities Act of 1933, the Securities Exchange Act of 1934, the Trust Indenture Act of 1939, and the Investment Company Act of 1940. EDGAR contains millions of company and individual filings. The system processes about 3,000 filings per day, serves up 3,000 terabytes of data to the public annually, and accommodates 40,000 new filers per year on average.

There are several ways to download the data, and some open source packages available to extract the text from these filings. However, these require extensive programming and are not always easy to use. Below we provide a simple one-API call that will create a dataset in a few lines of code, for any period of time and for a large number of tickers.

We have wrapped the extraction functionality into a SageMaker processing container and provide this notebook to enable users to download a dataset of filings with meta data such as dates and parsed plain text that can then be used for machine learning using other SageMaker tools. This is included in the [SageMaker Industry Jumpstart Industry](https://aws.amazon.com/blogs/machine-learning/use-pre-trained-financial-language-models-for-transfer-learning-in-amazon-sagemaker-jumpstart/) library for Financial language models. Users only need to specify a date range and a list of ticker symbols and this solution will do the rest.

As of now, the solution supports extracting a popular subset of SEC forms in plain text (excluding tables). These are 10-K, 10-Q, 8-K, 497, 497K, S-3ASR and N-1A. For each of these, we provide examples below and a brief description of each form. For the 10-K and 10-Q forms, filed every year or quarter, we also extract the Management Discussion and Analysis (MDNA) section, which is the primary forward-looking section in the filing. This is the section that has been most widely used in financial text analysis. Therefore, we provide this section automatically in a separate column of the dataframe alongside the full text of the filing.

The extracted dataframe is written to S3 storage and to the local notebook instance. 

### News Articles Dataset

We will use free [NewsCatcher API](https://docs.newscatcherapi.com/) to grab top 4-5 articles about the specific organization using filters, however other sources such as Social media feeds, RSS Feeds can also be used.

The first step in the pipeline is to fetch the SEC report fromt he EDGAR database using the [SageMaker Industry Jumpstart Industry](https://aws.amazon.com/blogs/machine-learning/use-pre-trained-financial-language-models-for-transfer-learning-in-amazon-sagemaker-jumpstart/) library for Financial language models. This library provides us an easy to use functionality to obtain either one or multiple SEC reports for one or more Ticker symbols or CIKs. The ticker or CIK number will be passed to the SageMaker Pipeline using Pipeline parameter `inference_ticker_cik`. For demo purposes of this Pipeline we will focus on a single Ticker/CIK number at a time and the MDNA section of the 10-K form. The first processing will extract the MDNA from the 10-K form for a company and will also gather a few news articles related to the company from the NewsCatcher API. This data will ultimately be used for summarization and then finally sentiment analysis.

We will setup the following SageMaker Pipeline. The Pipleline has two flows depending on what the value for `model_register_deploy` Pipeline parameter is set to. If the value is set to `Y` we want the pipeline to register the model and deploy the latest version of the model from the model registry to the SageMaker endpoint. If the value is set to `N` then we simply want to run inferences using the FinBert and the Pegasus models using the Ticker symbol (or CIK number) that is passed to the pipeline using the `inference_ticker_cik` Pipeline parameter.

1[end to end](./images/pipeline-success.png)