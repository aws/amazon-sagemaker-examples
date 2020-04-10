from datetime import datetime
import random

AUTOANNOTATION_THRESHOLD = 0.50
JOB_TYPE = "groundtruth/text-classification"

class SimpleActiveLearning:

    def __init__(self, job_name, label_category_name,
                  label_names, max_selections):
        self.job_name = job_name
        self.label_category_name = label_category_name
        self.label_names = label_names
        self.max_selections = max_selections

    def compute_margin(self, probabilities, labels):
      """
       compute the confidence and the best label given the probability distribution.
      """
      max_probability = max(probabilities)
      max_prob_index = probabilities.index(max_probability)
      best_label = labels[max_prob_index]
      remaining_probs = [prob for i,prob in enumerate(probabilities) if i!=max_prob_index]
      second_probability = max(remaining_probs, default=0.0)
      return max_probability - second_probability, best_label

    def get_label_index(self, inference_label_output):
        """
            inference_label_output is of the format "__label__0".
            This method gets an integer suffix from the end of the string.
            For this example, "__label__0" the function returns 0.
        """
        return int(inference_label_output.split('_')[-1])

    def make_metadata(self, margin, best_label):
      """
         make required metadata to match the output label.
      """
      return {
        'confidence': float(f'{margin: 1.2f}'),
        'job-name': self.job_name,
        'class-name': self.label_names[self.get_label_index(best_label)],
        'human-annotated': 'no',
        'creation-date': datetime.utcnow().strftime('%Y-%m-%dT%H:%m:%S.%f'),
        'type': JOB_TYPE
      }

    def make_autoannotation(self, prediction, source, margin, best_label):
      """
         generate the final output prediction with the label and confidence.
      """
      return {
        'source': source['source'],
        'id': prediction['id'],
        f'{self.label_category_name}': best_label,
        f'{self.label_category_name}-metadata': self.make_metadata(margin,
                      best_label)
      }

    def autoannotate(self, predictions, sources):
       """
         auto annotate all unlabeled data with confidence above AUTOANNOTATION_THRESHOLD.
       """
       sources_by_id = {
          source['id']: source for source in sources
       }
       autoannotations = []
       for prediction in predictions:
          probabilities = prediction['prob']
          labels = prediction['label']
          margin, best_label = self.compute_margin(probabilities, labels)
          if margin > AUTOANNOTATION_THRESHOLD:
            autoannotations.append(self.make_autoannotation(
                prediction, sources_by_id[prediction['id']],
                margin, best_label
            ))

       return autoannotations


    def select_for_labeling(self, predictions, autoannotations):
       """
         Select the next set of records to be labeled by humans.
       """
       initial_ids = {
          prediction['id'] for prediction in predictions
       }
       autoannotation_ids = {
          autoannotation['id'] for autoannotation in autoannotations
       }
       remaining_ids = initial_ids - autoannotation_ids
       selections = random.sample(
        remaining_ids, min(self.max_selections, len(remaining_ids))
       )
       return selections

