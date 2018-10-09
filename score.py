import os, sys
import re
from itertools import izip_longest
import random
import math
from bisect import bisect
from collections import namedtuple, defaultdict
import gzip
import json
from dateutil.parser import parse
import time
from datetime import datetime
import numpy as np
import pandas
from scipy.stats import itemfreq, norm, rankdata
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
# needed because of an rpy2 linking bug
import readline
import rpy2
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects as ro
import argparse
import warnings

# FINAL_LABELS_BASEDIR = "/mnt/data/TF_binding/DREAM_challenge/hidden_test_set_chipseq_data/tsvs" # whole genome

def optional_gzip_open(fname):
    return gzip.open(fname) if fname.endswith(".gz") else open(fname)  

# which measures to use to evaluate the overall score
MEASURE_NAMES = ['recall_at_10_fdr', 'recall_at_50_fdr', 'auPRC', 'auROC']
ValidationResults = namedtuple('ValidationResults', MEASURE_NAMES)

def recall_at_fdr(y_true, y_score, fdr_cutoff=0.05):
    # print y_true, y_score
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fdr = 1- precision
    cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
    return recall[cutoff_index]

def scikitlearn_calc_auPRC(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def calc_auPRC(y_true, y_score):
    """Calculate auPRC using the R package 

    """
    ro.globalenv['pred'] = y_score
    ro.globalenv['labels'] = y_true
    return ro.r('library(PRROC); pr.curve(scores.class0=pred, weights.class0=labels)$auc.davis.goadrich')[0]

class InputError(Exception):
    pass

ClassificationResultData = namedtuple('ClassificationResult', [
    'is_cross_celltype',
    'sample_type', # should be validation or test
    'train_chromosomes',
    'train_samples', 

    'validation_chromosomes',
    'validation_samples', 

    'auROC', 'auPRC', 'F1', 
    'recall_at_50_fdr', 'recall_at_25_fdr', 
    'recall_at_10_fdr', 'recall_at_05_fdr',
    'num_true_positives', 'num_positives',
    'num_true_negatives', 'num_negatives'])

class ClassificationResult(object):
    _fields = ClassificationResultData._fields

    def __iter__(self):
        return iter(getattr(self, field) for field in self._fields)

    def iter_items(self):
        return zip(self._fields, iter(getattr(self, field) for field in self._fields))
    
    def __init__(self, labels, predicted_labels, predicted_prbs,
                 is_cross_celltype=None, sample_type=None,
                 train_chromosomes=None, train_samples=None,
                 validation_chromosomes=None, validation_samples=None):
        # filter out ambiguous labels
        # JIN
        index = labels > -0.5
        predicted_labels = predicted_labels[index]
        predicted_prbs = predicted_prbs[index]
        labels = labels[index]


        self.is_cross_celltype = is_cross_celltype
        self.sample_type = sample_type

        self.train_chromosomes = train_chromosomes
        self.train_samples = train_samples

        self.validation_chromosomes = validation_chromosomes
        self.validation_samples = validation_samples
        
        positives = np.array(labels == 1)
        self.num_true_positives = (predicted_labels[positives] == 1).sum()
        self.num_positives = positives.sum()
        
        negatives = np.array(labels == 0)        
        self.num_true_negatives = (predicted_labels[negatives] == 0).sum()
        self.num_negatives = negatives.sum()

        if positives.sum() + negatives.sum() < len(labels):
            raise InputError("All labels must be either 0 or +1")
        
        try: self.auROC = roc_auc_score(positives, predicted_prbs)
        except ValueError: self.auROC = float('NaN')
        self.auPRC = calc_auPRC(positives, predicted_prbs)
        self.F1 = f1_score(positives, predicted_labels)
        self.recall_at_50_fdr = recall_at_fdr(
            labels, predicted_prbs, fdr_cutoff=0.50)
        self.recall_at_25_fdr = recall_at_fdr(
            labels, predicted_prbs, fdr_cutoff=0.25)
        self.recall_at_10_fdr = recall_at_fdr(
            labels, predicted_prbs, fdr_cutoff=0.10)
        self.recall_at_05_fdr = recall_at_fdr(
            labels, predicted_prbs, fdr_cutoff=0.05)
        return

    @property
    def positive_accuracy(self):
        return float(self.num_true_positives)/(1e-6 + self.num_positives)

    @property
    def negative_accuracy(self):
        return float(self.num_true_negatives)/(1e-6 + self.num_negatives)

    @property
    def balanced_accuracy(self):
        return (self.positive_accuracy + self.negative_accuracy)/2    

    def iter_numerical_results(self):
        for key, val in self.iter_items():
            try: _ = float(val) 
            except TypeError: continue
            yield key, val
        return

    def __str__(self):
        rv = []
        if self.train_samples is not None:
            rv.append("Train Samples: %s\n" % self.train_samples)
        if self.train_chromosomes is not None:
            rv.append("Train Chromosomes: %s\n" % self.train_chromosomes)
        if self.validation_samples is not None:
            rv.append("Validation Samples: %s\n" % self.validation_samples)
        if self.validation_chromosomes is not None:
            rv.append("Validation Chromosomes: %s\n" % self.validation_chromosomes)
        rv.append("Bal Acc: %.3f" % self.balanced_accuracy )
        rv.append("auROC: %.3f" % self.auROC)
        rv.append("auPRC: %.3f" % self.auPRC)
        rv.append("F1: %.3f" % self.F1)
        rv.append("Re@0.50 FDR: %.3f" % self.recall_at_50_fdr)
        rv.append("Re@0.25 FDR: %.3f" % self.recall_at_25_fdr)
        rv.append("Re@0.10 FDR: %.3f" % self.recall_at_10_fdr)
        rv.append("Re@0.05 FDR: %.3f" % self.recall_at_05_fdr)
        rv.append("Positive Accuracy: %.3f (%i/%i)" % (
            self.positive_accuracy, self.num_true_positives,self.num_positives))
        rv.append("Negative Accuracy: %.3f (%i/%i)" % (
            self.negative_accuracy, self.num_true_negatives, self.num_negatives))
        return "\t".join(rv)

def build_ref_scores_array(ref_fname):
    factor = os.path.basename(ref_fname).split(".")[0]
    
    with gzip.open(ref_fname) as fp:
        header_data = fp.readline().split()
        sample_names = header_data[3:]
    samples = {}
    df = pandas.read_csv(ref_fname, sep="\t")
    # filter the data frame by the test regions
    contig_mask = ( (df['chr'].values == 'chr1') 
                    | (df['chr'].values == 'chr8') 
                    | (df['chr'].values == 'chr21') )
    # JIN
    for sample_name in sample_names:
        char_data = df[sample_name].values[contig_mask]
        samples[sample_name] = np.zeros(char_data.shape, dtype=int)
        samples[sample_name][char_data == 'B'] = 1
        samples[sample_name][char_data == 'A'] = -1
    return samples

def build_submitted_scores_array(fname):
    df = pandas.read_csv(
        fname, names=["chr", "start", "stop", "sample"], sep="\t")
    # return df['sample'].values
    df2 = df[df.chr.str.contains(r'^chr(1|8|21)$',regex=True)]
    return df2['sample'].values

def score_final_main(submission_fname,label_dir):
    # load and parse the submitted filename
    fname_pattern = "^[FBL]\.(.+?)\.(.+?)\.tab.gz"
    res = re.findall(fname_pattern, os.path.basename(submission_fname))
    if len(res) != 1 or len(res[0]) != 2:
        raise InputError, "The submitted filename ({}) does not match expected naming pattern '{}'".format(
            submission_fname, fname_pattern)
    else:        
        factor, cell_line = res[0]
    # find a matching validation file
    labels_fname = os.path.join(label_dir, "{}.train.labels.tsv.gz".format(factor))
    # validate that the matching file exists and that it contains labels that 
    # match the submitted sample 
    header_data = None
    try:
        with gzip.open(labels_fname) as fp:
            header_data = next(fp).split()
    except IOError:
        raise InputError("The submitted factor, sample combination ({}, {}) is not a valid final submission.".format(
            factor, cell_line))
    # Make sure the header looks right
    assert header_data[:3] == ['chr', 'start', 'stop']
    labels_file_samples = header_data[3:]
    # We only expect to see one sample per leaderboard sample
    if cell_line not in labels_file_samples:
        raise InputError("The submitted factor, sample combination ({}, {}) is not a valid final submission.".format(
            factor, cell_line))
    try:
        scores = build_submitted_scores_array(submission_fname)
        labels = build_ref_scores_array(labels_fname)[cell_line]
    # JIN
        full_results = ClassificationResult(labels, scores.round(), scores)
    except:
        print "ERROR", submission_fname
        sys.stdout.flush()

    return full_results, labels, scores
    # print scores

def main():
    parser = argparse.ArgumentParser(prog='Scoring script for ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge', 
        description='''
        For leaderboard, use label at Files/Challenge Resources/scoring_script/labels/leaderboard.
        For final/within, use label at Files/Challenge Resources/scoring_script/labels/final.
        ''')
    parser.add_argument('submission_fname', metavar='submission_fname', type=str,
                            help='Submission file name should match with pattern [L/F/B].[TF_NAME].[CELL_TYPE].gz.')
    parser.add_argument('label_dir', metavar='label_dir', type=str,
                            help='Directory containing label files. Label files should have a format of [TF_NAME].train.labels.tsv.gz. Download at Files/Challenge Resources/scoring_script/labels')
   
    args = parser.parse_args()    
    warnings.simplefilter(action='ignore', category=UserWarning)
    results, labels, scores = score_final_main(args.submission_fname,args.label_dir)
    print results

if __name__ == '__main__':
    main()

'''
python score.py ~/F.HNF4A.liver.tab.gz /mnt/data/TF_binding/DREAM_challenge/all_labels/final
python score.py ~/B.HNF4A.liver.tab.gz /mnt/data/TF_binding/DREAM_challenge/all_labels/within
python score.py ~/L.CTCF.GM12878.tab.gz /mnt/data/TF_binding/DREAM_challenge/all_labels/leaderboard
'''
