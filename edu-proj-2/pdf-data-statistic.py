import os
import string
import sys
import argparse
import time

import nltk
import itertools
from collections.abc import Iterable
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar

# argument setting
parser = argparse.ArgumentParser(description='PDF data statistic')
parser.add_argument('act', type=str, help='actions, e.g.:freq, occur, download',
                    choices=["freq", "occur", "col", "download"])
parser.add_argument('--pdf', type=str, nargs='+', help='pdf files', default=[])
parser.add_argument('--path', type=str, nargs='+', help='pdf file path', default=[])
parser.add_argument('--headers', metavar='N', type=str, nargs='+', default=['findings', 'results'], help='headers')
parser.add_argument('--keywords', metavar='N', type=str, nargs='+', default=[],
                    help='keywords，empty keywords means all')
parser.add_argument('--top', type=int, default=100, help='top count')
parser.add_argument('--verbose', type=bool, default=False, action=argparse.BooleanOptionalAction, help='verbose')
args = parser.parse_args()


class PDFAnalyzer:
    def __init__(self, extractor, statistic):
        self.extractor = extractor
        self.statistic = statistic

    def freq(self, pdfs, headers, keywords, top):
        # find all contents
        contents = ""
        for pdf in pdfs:
            content = self.extractor.extract_specific_abstract_content(pdf)
            contents += " " + content
            if args.verbose:
                print(content)

            # calculate frequency
            print("====== Top {} frequency: {} ======".format(top, pdf))
            keywords_appear = self.statistic.keywords(content, keywords)
            dist = nltk.FreqDist(keywords_appear)
            keywords_top = dist.most_common(top)
            print(keywords_top)

        # calculate frequency
        print("====== Top {} frequency total ======".format(top))
        keywords_appear = self.statistic.keywords(contents, keywords)
        dist = nltk.FreqDist(keywords_appear)
        keywords_top = dist.most_common(top)
        print(keywords_top)
        return

    def occur(self, pdfs, headers, keywords, top):
        # find all contents
        contents = ""
        for pdf in pdfs:
            content = self.extractor.extract_specific_abstract_content(pdf)
            contents += " " + content

            if args.verbose:
                print(content)

            # calculate occurring
            print("====== Top {} occurring frequency: {} ======".format(top, pdf))
            keywords_group = self.statistic.keywords_group(content, keywords)
            dist = nltk.FreqDist(keywords_group)
            keywords_group_top = dist.most_common(top)
            print(keywords_group_top)

        # calculate occurring
        print("====== Top {} occurring frequency total ======".format(top))
        keywords_group = self.statistic.keywords_group(contents, keywords)
        dist = nltk.FreqDist(keywords_group)
        keywords_group_top = dist.most_common(top)
        print(keywords_group_top)
        return

    def collocation(self, pdfs, headers, keywords, top):
        # find all contents
        contents = ""
        for pdf in pdfs:
            if args.verbose:
                print("====== PDF: {} ======".format(pdf))

            content = self.extractor.extract_specific_abstract_content(pdf)
            contents += " " + content

            if args.verbose:
                print(content)

            # calculate occurring
            print("====== Top {} collocation frequency: {} ======".format(top, pdf))
            finder = nltk.collocations.TrigramCollocationFinder.from_words(
                self.statistic.keywords(content, keywords), window_size=3)
            sorted_freq_dist = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:top]
            print(sorted_freq_dist)

        # calculate occurring
        print("====== Top {} collocation frequency total ======".format(top))
        # print("content: {}".format(self.statistic.keywords(contents, keywords)))
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        finder = nltk.collocations.TrigramCollocationFinder.from_words(
            self.statistic.keywords(contents, keywords), window_size=3)
        # colls = finder.nbest(trigram_measures.likelihood_ratio, top)
        # print(list(nltk.trigrams(self.statistic.keywords(contents, keywords))))
        # print(colls)
        sorted_freq_dist = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:top]
        print(sorted_freq_dist)


class Extractor:
    bolds = ["Bold", "bold"]
    bolds_endwith = [".B", "B"]

    def extract_element_font_and_size(self, element):
        for text_line in element:
            if not text_line:
                continue
            if not isinstance(text_line, Iterable):
                continue
            for character in text_line:
                if character is None:
                    continue
                if isinstance(character, LTChar):
                    bold = False
                    for b in Extractor.bolds:
                        if character.fontname.find(b) != -1:
                            bold = True
                    for be in Extractor.bolds_endwith:
                        if character.fontname.endswith(be):
                            bold = True
                    return character.fontname, round(character.size, 4), bold
        return "", 0, False

    def extract_specific_header_content(self, pdf, headers):
        header_found = False
        ending_found = False
        font = ""
        size = ""
        bold = False
        content = ""

        first_page = True
        for page_layout in extract_pages(pdf):
            if first_page:
                first_page = False
                continue

            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    if not header_found:
                        # convert element text to lower case
                        lowercase = element.get_text().lower()
                        if len(lowercase) > 17:
                            lowercase = element.get_text()[0:17].lower()

                        # check if element contains header string, e.g.: "findings", "results"
                        for header in headers:
                            # search for header
                            if lowercase.find(header) == -1:
                                continue

                            # get font and size of header
                            font, size, bold = self.extract_element_font_and_size(element)
                            # print("header: ", lowercase, font, size, bold)
                            if bold:
                                header_found = True
                                break

                    # if header not found, continue searching
                    if not header_found:
                        continue

                    # print("====== element ======")
                    # print(element.get_text())

                    # if content length is zero, concatenate element text and continue
                    if len(content) == 0:
                        content += element.get_text()
                        continue

                    # if header found, and content length is not zero, concatenate element text to content
                    # get element font & size
                    cur_font, cur_size, cur_bold = self.extract_element_font_and_size(element)
                    # print("======", cur_font, cur_size, cur_bold, element.get_text())
                    if cur_font == font and cur_size == size and bold:
                        ending_found = True
                        break

                    content += element.get_text()
            if ending_found:
                break
        return content

    def extract_specific_abstract_content(self, pdf):
        content = ""

        # find paragraph with prefix of ABSTRACT
        for page_layout in extract_pages(pdf):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    paragraph = element.get_text().lstrip(" ")
                    paragraph = " ".join(paragraph.split())
                    if paragraph.startswith("ABSTRACT"):
                        content += paragraph
                        break
            if len(content) != 0:
                break

        # if content not found, the take the first paragraph that word length is greater than 50
        if len(content) != 0:
            return content

        for page_layout in extract_pages(pdf):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    paragraph = element.get_text().lstrip(" ")
                    paragraph = " ".join(paragraph.split())
                    words = paragraph.split(" ")
                    if len(words) > 50:
                        content += paragraph
                        break
            if len(content) != 0:
                break

        return content


class Statistic:
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

    def tokenize(self, content):
        return nltk.word_tokenize(content)

    def sent_tokenize(self, content):
        return nltk.sent_tokenize(content, language='english')

    def remove_stop_words(self, words):
        words_without_stopwords = []
        stop_words = set(nltk.corpus.stopwords.words('english'))
        for word in words:
            if word in stop_words:
                continue
            if word in string.punctuation:
                continue
            if len(word) == 1:
                continue
            words_without_stopwords.append(word)
        return words_without_stopwords

    def lemmatize(self, words):
        words_after_lem = []
        lem = nltk.stem.WordNetLemmatizer()
        for word in words:
            words_after_lem.append(lem.lemmatize(word))
        return words_after_lem

    def filter_pos_tags(self, words, allowed_tags):
        # pos tag
        word_tags = nltk.pos_tag(words)

        # filter tags
        words_after_filter = []
        for word, tag in word_tags:
            if tag in allowed_tags:
                words_after_filter.append(word)
        return words_after_filter

    def tolower(self, words):
        words_lower = []
        for word in words:
            words_lower.append(word.lower())
        return words_lower

    def keywords(self, content, keywords):
        # word tokenization
        words = self.tokenize(content)

        # remove stop words
        words_without_stopwords = self.remove_stop_words(words)

        # lemmatize
        words_after_lem = self.lemmatize(words_without_stopwords)
        words_after_lem = self.tolower(words_after_lem)
        keywords_after_lem = self.lemmatize(keywords)
        keywords_after_lem = self.tolower(keywords_after_lem)

        # filter keywords
        words_filtered = []
        if len(keywords) != 0:
            keywords_set = set(keywords_after_lem)
            words_filtered = []
            for word in words_after_lem:
                if word in keywords_set:
                    words_filtered.append(word)
        else:
            words_filtered = words_after_lem

        # pos tagging & filtering
        tags = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        return self.filter_pos_tags(words_filtered, tags)

    def keywords_group(self, content, keywords):
        # sentence tokenization
        sents = self.sent_tokenize(content)

        # prepare grouped sentences
        sent_groups = []
        for index in range(0, len(sents)):
            if index == 0 and len(sents) == 1:
                sent_groups.append([sents[index]])
            elif index == 0 and len(sents) > 1:
                sent_groups.append([sents[index], sents[index+1]])
            elif index == len(sents) - 1:
                sent_groups.append([sents[index-1], sents[index]])
            else:
                sent_groups.append([sents[index-1], sents[index], sents[index+1]])

        # extract word occurring
        combs = []
        for group in sent_groups:
            # concatenate sentences to string
            sent_content = ''.join(group)

            # pre-process sentence content
            keywords_appear = self.keywords(sent_content, keywords)

            # remove duplicate keywords
            keywords_found = list(dict.fromkeys(keywords_appear))

            # generate combinations, the maximum repeat is 3
            repeat = 3
            if len(keywords_found) < 3:
                repeat = len(keywords_found)
            group_combs = itertools.combinations(keywords_found, repeat)
            for group_comb in group_combs:
                sorted_group_comb = list(group_comb)
                sorted_group_comb.sort()
                if len(sorted_group_comb) == 0:
                    continue
                combs.append('|'.join(sorted_group_comb))
        return combs


def main():
    for p in args.path:
        if args.path is not None:
            r = os.walk(p)
            for path, dirs, names in r:
                for name in names:
                    if name.startswith("."):
                        continue
                    args.pdf.append(os.path.join(path, name))
    # make action
    match args.act:
        case 'freq':
            analyzer = PDFAnalyzer(extractor=Extractor(), statistic=Statistic())
            analyzer.freq(args.pdf, args.headers, args.keywords, args.top)
        case 'occur':
            analyzer = PDFAnalyzer(extractor=Extractor(), statistic=Statistic())
            analyzer.occur(args.pdf, args.headers, args.keywords, args.top)
        case 'col':
            analyzer = PDFAnalyzer(extractor=Extractor(), statistic=Statistic())
            analyzer.collocation(args.pdf, args.headers, args.keywords, args.top)
        case 'download':
            nltk.download()


if __name__ == '__main__':
    main()

