
import sacrebleu
import mauve
from sacrebleu.metrics import BLEU, CHRF, TER
from rouge import Rouge
import nltk
from nltk.translate import meteor
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.corpus import wordnet

def compute_llm_metrics(reference, candidate):
  bleu_mean = calculate_bleu_score(reference, candidate)
  chrf_mean = calculate_chrf_score(reference, candidate)
  ter_mean = calculate_ter_score(reference, candidate)
  #meteor_mean = calculate_meteor_score(reference, candidate)
  
  results = {"BLEU": bleu_mean, "CHRF": chrf_mean, "TER": ter_mean, }

  return results

# def calculate_meteor_score(reference, candidate):
#   reference = word_tokenize([reference])
#   candidate = word_tokenize([candidate])
#   score = round(meteor_score([reference],candidate), 4)

#   return score

def calculate_ter_score(reference, candidate):
  ter = TER()
  score = ter.corpus_score([candidate], [[reference]])

  return score.score

def calculate_bleu_score(reference, candidate):
  bleu = BLEU()
  score = bleu.corpus_score([candidate], [[reference]])
  
  return score.score

def calculate_chrf_score(reference, prediction):
  chrf = CHRF()
  score = chrf.corpus_score([prediction], [[reference]])

  return score.score