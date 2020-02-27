from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autolearn.autolearn import AutoLearn


class AutoPipeline:
    def __init__(self,
                 df,
                 target,
                 subtables=None,
                 autoconverter=None,
                 autolearn=None):
        """[Draft] AutoPipleline aggregates several steps into a single step.

        Note that the class is still in the draft stage, API may change.

        Args:
            df (pd.DataFrame): main table
            target (str): target value
            subtables (dict): See AutoConverter.fit() API documentation
            autoconverter (ritml.AutoConverter):
            autolearn (ritml.AutoLearn):

        """
        self.df = df
        self.target = target
        self.subtables = subtables

        if autolearn is None:
            # If not specified, use AutoLearn with default settings
            self.al = AutoLearn()
        else:
            self.al = autolearn

        if autoconverter is None:
            # If not specified, use AutoConverter with default settings
            self.ac = AutoConverter(target=self.target)
        else:
            self.ac = autoconverter

    def run(self):
        X, y = self.ac.fit_transform(self.df,
                                     subtables=self.subtables)
        self.results = self.al.learn(X, y)
