import pandas as pd

from learnit.autopipeline.autopipeline import AutoPipeline

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    pipeline = AutoPipeline(df, target="Survived")
    pipeline.run()
    results = pipeline.results
    print(results['name'])
    print(results['eval_df'])
