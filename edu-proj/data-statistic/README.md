# Introduction
This project is to help analyze data of pdf file and generate statistic.

# Requirement
``` python
    # To begin this process, we agreed on two initial pre-processing approaches:
    #   1. Keyword frequency, and
    #   2. Words frequently occurring together in sentences (e.g. groups of words occurring in a sentence,
    #       along with the sentence before and the sentence after).
    #       a. Keyword frequency – We agreed to look at nouns, adjectives, adverbs and roots.
    #           We would like an initial list of the top 100 frequently occurring words.
    #           Tags: ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    #       b. Words occurring together – We agreed to start with the top 30 groups of words occurring together
    #           in sentences. Also, which articles these groups of words appear
    #           (e.g. Word group 1 occurs in Paper 1 & 6).
```

# Pre-requisite
## pdfminer installation
[Refer to Github](https://github.com/pdfminer/pdfminer.six)
## NLTK installation
[Refer to NLTK official website](https://www.nltk.org/install.html)
## NLTK datasets/models downloading
Refer to NLTK downloading in the next section.
* Step 1. run the download command.
* Step 2. click 'download' button to continue, it may take several minutes to tens of minutes depend on the network.

# Command line
## NLTK downloading
```shell
> python pdf-data-statistic.py download
```

## Keywords Frequency
### Usage Example
```shell
> python pdf-data-statistic.py freq --pdf ./Tondeur\ et\ al\ 2013.pdf ./Butler\ \&\ Leahy\ 2020.pdf --headers findings results --keywords finding present ict tpack --top 50
```

## Keywords Occurring
### Usage Example
```shell
> python pdf-data-statistic.py occur --pdf ./Tondeur\ et\ al\ 2013.pdf ./Butler\ \&\ Leahy\ 2020.pdf --headers findings results --keywords finding present ict tpack --top 50
```

## Options
```
--pdf         => PDF file list to be processed
--headers     => header names for searching
--keywords    => keywords list used in statistic
--top         => numbers of top results returned, for 'freq' return top keywords, for 'occur' return top keywords group
--verbose     => more detailed output will be printed
```