import pandas as pd
from sklearn.cluster import KMeans
from pre_model import df

from sklearn.metrics import silhouette_score

def create_model():
    features  = ["income", "age"]
    model = KMeans(n_clusters=4, n_init="auto")

    values= model.fit_predict(df[features])

    predictions = pd.DataFrame(values, columns=["predicted_cluster"])


    result_df = pd.concat(   [df, predictions], axis=1    )

    result_df["predicted_cluster"] = result_df["predicted_cluster"].astype("object")


    ans = silhouette_score(df[features],df["predicted_cluster"] )
    print(f"Silhouette score: {ans}" )