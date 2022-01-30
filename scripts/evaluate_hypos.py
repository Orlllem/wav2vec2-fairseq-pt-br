import json
import logging
import os
import sys

from datasets import load_metric

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("prepare_data")

wer = load_metric("wer")
cer = load_metric("cer")

f = open('reports/hypo.word-checkpoint_best.pt-test.txt', 'r')
file_contents = f.read()

predictions = file_contents.split("\n")[:-1]
predictions = [i.split(" (")[0] for i in predictions]


f = open('reports/ref.word-checkpoint_best.pt-test.txt', 'r')
file_contents = f.read()

references = file_contents.split("\n")[:-1]

references = [i.split(" (")[0] for i in references]

wer_compute = wer.compute(predictions=predictions,
                          references=references) * 100

cer_compute = cer.compute(predictions=predictions,
                          references=references) * 100

logger.info("WER: %f", wer_compute)
logger.info("CER: %f", cer_compute)

results = {"WER": str(wer_compute), "CER": str(cer_compute)}

with open("reports/metrics.json", 'w') as output_file:
    json.dump(results, output_file)
