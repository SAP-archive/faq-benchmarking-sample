import argparse
import json
import logging
import string
import sys

import pandas as pd
import regex as re
import requests
from tqdm import tqdm

logFormatter = logging.Formatter(
    "%(asctime)-40s %(funcName)-25s %(levelname)-15s  %(message)-25s")
LOGGER = logging.getLogger()

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
LOGGER.addHandler(consoleHandler)

LOGGER.setLevel(logging.INFO)

API_ENDPOINT = 'https://api.cai.tools.sap/train/v2/request/ask'


class Client:

    def __init__(self, url, request_token, language):
        self.url = url
        self.headers = {"Authorization": "Token {}".format(request_token)}
        self.language = language

    def _prepare_payload(self, text):
        return {"text": text, "number_of_answers": 3, "language": self.language}

    def connect(self):
        try:
            test_payload = self._prepare_payload("test")
            response = self.call_api(test_payload)
            if response.status_code == 200:
                LOGGER.info("Connection to API successful: {}".format(response))
            else:
                LOGGER.error("There was an error connecting to the API: {}".format(response))
                LOGGER.error(response.text)
                sys.exit(1)
        except Exception as e:
            LOGGER.error(e)

    def call_api(self, payload):
        response = requests.post(self.url, headers=self.headers, data=payload)
        return response


class Benchmark(Client):

    def __init__(self, url, request_token, csv_file, language="en"):
        super().__init__(url, request_token, language)
        self.connect()
        self.data_df, self.questions, self.ground_truth_answers = self._load_csv(csv_file)
        self.preprocessed_answers = {}

    def _load_csv(self, csv_file):
        try:
            data = pd.read_csv(csv_file)
            if set(data.columns) == {'question', 'answer'}:
                LOGGER.info("{} question/answer pairs loaded from CSV successfully.".format(len(data)))
                return data, data['question'], data['answer']
            else:
                LOGGER.error("Dataset must have 'question' and 'answer' as CSV headers")
                sys.exit(1)

        except Exception as e:
            LOGGER.error(e)
            sys.exit(1)

    def _preprocess(self, query):
        try:
            if query in self.preprocessed_answers:
                return self.preprocessed_answers[query]
            else:
                preprocessed_query = re.sub(r'[\p{P}\p{S}\s]', '', query.lower())
                self.preprocessed_answers[query] = preprocessed_query
                return preprocessed_query

        except Exception:
            return query

    def predict(self, questions):
        raw_predictions, top_3_sets = [], []

        try:
            print("\n")
            for question in tqdm(questions):
                payload = self._prepare_payload(question)
                response = self.call_api(payload)
                result_json = json.loads(response.text)

                top_3_answers = [res['answer'] for res in result_json['results']['faq']]
                preprocessed_top_3 = [self._preprocess(ans) for ans in top_3_answers]
                top_3_answer_sets = [set(preprocessed_top_3[:i]) for i in range(1, 4)]

                raw_predictions.append(top_3_answers)
                top_3_sets.append(top_3_answer_sets)
            print("\n")
            return raw_predictions, top_3_sets

        except Exception as e:
            LOGGER.error(e)

    def analyze_results(self, questions, true_answers, top_3_sets):
        hits = [0] * 3
        not_answered = [set(), set(), set()]

        for i, ans in enumerate(true_answers):
            for j, preds in enumerate(top_3_sets[i]):
                if ans in preds:
                    hits[j] += 1
                else:
                    not_answered[j].add(questions[i])

        return hits, not_answered

    def generate_report(self):

        LOGGER.info("Calling the API endpoint to fetch answers from QnA bot...")
        raw_predictions, top_3_sets = self.predict(self.questions)
        self.data_df['predictions'] = raw_predictions

        LOGGER.info("Normalizing ground truth answers - removing punctuation, whitespace and special chars.")
        preprocessed_answers = [self._preprocess(ans) for ans in self.ground_truth_answers]

        hits, not_answered = self.analyze_results(self.questions, preprocessed_answers, top_3_sets)
        n = len(self.questions)

        for i in range(3):
            print("\n", "*" * 100, "\n")
            LOGGER.info("Top {} Accuracy: {} %".format(i + 1, round(hits[i] / n*100, 2)))
            LOGGER.info(
                "Number of questions answered incorrectly by top {} predicted answer(s) - {}".format(i + 1, n - hits[i]))
            print("These questions need attention -- ",  not_answered[i])
            print("\n")


if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(
        prog='benchmark.py',
        description="Allows benchmarking of gold dataset performance in a QnA bot"
    )
    parser.add_argument(
        '--csv_file',
        dest='csv_file',
        metavar='CSV',
        type=str,
        help='CSV file with [question, answer] header',
        required=True
    )
    parser.add_argument(
        '--request_token',
        dest='request_token',
        metavar='REQUEST_TOKEN',
        type=str,
        help='The Request Token can be found in the bot settings on the platform.',
        required=True
    )
    parser.add_argument(
        '--language',
        dest='language',
        metavar='LANGUAGE',
        default='en',
        type=str,
        help='the predominant language of the questions and answers'
    )
    args = parser.parse_args()
    b = Benchmark(url=API_ENDPOINT,
                  request_token=args.request_token, csv_file=args.csv_file, language=args.language)

    b.generate_report()
