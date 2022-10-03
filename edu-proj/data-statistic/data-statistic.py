import sys
import argparse
import nltk
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar

# argument setting
parser = argparse.ArgumentParser(description='PDF data statistic')
parser.add_argument('act', type=str, help='actions, e.g.:run, download', choices=["run", "download"])
parser.add_argument('--pdf', metavar='N', type=str, nargs='+', help='pdf files')
parser.add_argument('--headers', metavar='N', type=str, nargs='+', default=['findings', 'results'], help='headers')
parser.add_argument('--keywords', metavar='N', type=str, nargs='+', help='keywords')
parser.add_argument('--top', type=int, default=100, help='top count')
parser.add_argument('--verbose', type=bool, default=False, action=argparse.BooleanOptionalAction, help='verbose')
args = parser.parse_args()

class PDFAnalyzer:
    def __init__(self, extractor, statistic):
        self.extractor = extractor
        self.statistic = statistic

    def run(self, pdfs, headers, keywords, top):
        # find all contents
        contents = ""
        for pdf in pdfs:
            content = self.extractor.extract_specific_header_content(pdf, headers)
            contents += " " + content

            if args.verbose:
                print("====== PDF: {} ======".format(pdf))
                print(content)

        # calculate frequency
        print("====== Top 100 frequency ======")
        freq_dist = self.statistic.keyword_freq(contents, keywords, top)
        print(freq_dist)


class Extractor:
    def extract_element_font_and_size(self, element):
        # get font and size of the first char of element
        for text_line in element:
            for character in text_line:
                if isinstance(character, LTChar):
                    return character.fontname, character.size

    def extract_specific_header_content(self, pdf, headers):
        header_found = False
        ending_found = False
        font = ""
        size = ""
        content = ""

        for page_layout in extract_pages(pdf):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    if not header_found:
                        # convert element text to lower case
                        lowercase = element.get_text().lower()
                        if len(lowercase) > 13:
                            lowercase = element.get_text()[0:13].lower()

                        # check if element contains header string, e.g.: "findings", "results"
                        for header in headers:
                            # search for header
                            if lowercase.find(header) == -1:
                                continue
                            header_found = True

                            # get font and size of header
                            font, size = self.extract_element_font_and_size(element)
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
                    cur_font, cur_size = self.extract_element_font_and_size(element)
                    if cur_font == font and cur_size == size:
                        ending_found = True
                        break

                    content += element.get_text()
            if ending_found:
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

    def remove_stop_words(self, words):
        words_without_stopwords = []
        stop_words = set(nltk.corpus.stopwords.words('english'))
        for word in words:
            if word in stop_words:
                continue
            words_without_stopwords.append(word)
        return words_without_stopwords

    def stemming(self, words):
        words_after_stemming = []
        porter_stemming = nltk.stem.PorterStemmer()
        for word in words:
            words_after_stemming.append(porter_stemming.stem(word))
        return words_after_stemming

    def filter_pos_tags(self, words, allowed_tags):
        # pos tag
        word_tags = nltk.pos_tag(words)

        # filter tags
        word_tags_after_filter = []
        for word, tag in word_tags:
            if tag in allowed_tags:
                word_tags_after_filter.append(word)
        return word_tags_after_filter

    def tolower(self, words):
        words_lower = []
        for word in words:
            words_lower.append(word.lower())
        return words_lower

    def keyword_freq(self, content, keywords, top):
        # tokenization
        words = self.tokenize(content)

        # remove stop words
        words_without_stopwords = self.remove_stop_words(words)

        # stemming
        words_after_stemming = self.stemming(words_without_stopwords)
        words_after_stemming = self.tolower(words_after_stemming)
        keywords_after_stemming = self.stemming(keywords)
        keywords_after_stemming = self.tolower(keywords_after_stemming)

        # filter keywords
        keywords_set = set(keywords_after_stemming)
        words_filtered = []
        for word in words_after_stemming:
            if word in keywords_set:
                words_filtered.append(word)

        # pos tagging & filtering
        tags = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        word_tags_after_filter = self.filter_pos_tags(words_filtered, tags)

        dist = nltk.FreqDist(word_tags_after_filter)
        return dist.most_common(top)


def main():
    # make action
    match args.act:
        case 'run':
            analyzer = PDFAnalyzer(extractor=Extractor(), statistic=Statistic())
            analyzer.run(args.pdf, args.headers, args.keywords, args.top)
        case 'download':
            nltk.download()


if __name__ == '__main__':
    main()

