# Bring Your Own Model (k-means)

*kmeans_bring_your_own_model.ipynb:* shows how to fit a k-means model in scikit-learn and then inject it into Amazon SageMaker's first party k-means container for scoring.  This addresses the use case where a customer has already trained their model outside of Amazon SageMaker, but wishes to host it for predictions within Amazon SageMaker.
