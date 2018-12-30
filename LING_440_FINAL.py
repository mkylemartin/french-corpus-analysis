# coding: utf-8

# # Read in Corpus

# In[1]:


corpus_path = ('/Users/Kyle/Documents/Box Sync/old/'
               'MARTIN-french-corpus-github/'
               '5_french-news-corpus.txt')
RAW_CORPUS = open(corpus_path, 'r').read()
print(len(RAW_CORPUS))  # how much did I start with?


# ## Remove non-interesting information
#
# For better n-grams!
#
# 1. Remove a `cookie_message` on all of the sites that sort of messes with our
#    ability to pull meaningful n-grams out of the corpus.
#
# 2. Remove a slogan (`lemonde_slogan`) that's at the bottom of all articles by
#    [lemonde.fr](https://lemonde.fr)
#
# 3. Related info message (`lemonde_info`) also removed

# In[2]:


import re
undesirables = {}
undesirables['cookie_message'] = ("En poursuivant votre navigation sur ce site, "
                                  "vous acceptez nos CGV et l\’utilisation de cookies "
                                  "pour vous proposer des contenus et services adaptés"
                                  " à vos centres d’intérêts et vous permettre "
                                  "l\'utilisation de boutons de partages sociaux."
                                  r"\s+En savoir plus et gérer ces paramètres\.")

undesirables['lemonde_slogan'] = (r"Découvrez chaque jour toute l\'info en\s+"
                                  r"direct \(de la politique à l\'économie en "
                                  r"passant par le sport et la\s+"
                                  r"météo\) sur Le Monde\.fr, le site de news "
                                  r"leader de la presse française\s+"
                                  r"en ligne\.")

undesirables['lemonde_info'] = ("Journal d\'information en ligne, "
                                "Le Monde.fr offre à ses visiteurs "
                                r"un\s+panorama complet de l\'actualité.")


# In[3]:


def remove_undesirables(message, name, corpus):
    """ receives a dictionary organized by
        phenom and message to be removed. """
    amount = len(re.findall(message, corpus, flags=re.IGNORECASE))
    filtered_corpus = re.sub(message,  # to be subbed
                             "",  # remove this
                             corpus,  # current state of corpus
                             flags=re.IGNORECASE)  # just to be sure
    print(f'removed {amount} instances of {name} `{message[:10]}(...)`')

    return filtered_corpus


# In[4]:


len(re.findall(undesirables['lemonde_info'], RAW_CORPUS, flags=re.IGNORECASE))


# ## Corpus filtration pipeline

# In[5]:


def filtration_pipeline(RAW_CORPUS):
    """ filters the corpus removing the boogers for a breath of fresh air """

    no_cookies = remove_undesirables(undesirables['cookie_message'],
                                     'cookie_message',
                                     RAW_CORPUS)

    no_lemonde = remove_undesirables(undesirables['lemonde_slogan'],
                                     'lemonde_slogan',
                                     no_cookies)

    no_info = remove_undesirables(undesirables['lemonde_info'],
                                  'lemonde_info',
                                  no_lemonde)

    filtered = no_info

    return filtered


# In[6]:


raw_filtered = filtration_pipeline(RAW_CORPUS)


# # Post-baptism n-grams
#
# To complete this aspect of the project I will use the NLTK n-gram feature.
# I will first tokenize all of the text. Then I will pickle the tokens so that
# I can avoid reprocessing the mass of text each time I run the script.
# There is a simple implementation of n- grams I used in a previous
# assignment, and will be able to just reduplicate that code for this step.

# In[7]:


from collections import Counter

from nltk.util import ngrams
from nltk.tokenize import word_tokenize


# ## `nltk.word_tokenize` everything

# In[8]:


nltk_tokens = word_tokenize(raw_filtered)


# ## Calculate n-grams

# In[9]:


bigrams = list(Counter(ngrams(nltk_tokens, 2)).items())
trigrams = list(Counter(ngrams(nltk_tokens, 3)).items())
fourgrams = list(Counter(ngrams(nltk_tokens, 4)).items())
fivegrams = list(Counter(ngrams(nltk_tokens, 5)).items())

grams = [bigrams, trigrams, fourgrams, fivegrams]


# In[10]:


i = 2
for gram in grams:
    print(f'{i}-grams')
    print(len(gram), 'grams')
    i += 1


# ## Display Results

# In[11]:


len(set(fourgrams))


# In[12]:


sorted(fivegrams, key=lambda x: x[1], reverse=True)[:1000]


# # Markov Chains
#
# I will pass my corpus through
# [`markovify`](https://github.com/jsvine/markovify).

# In[13]:


import markovify


# In[14]:


text_model = markovify.Text(raw_filtered)


# In[15]:


for i in range(10):
    print(text_model.make_sentence(tries=100))
    print()


# # Collocates
# Finding differences in collocates in four
# different words that mean 'exit' in French.

# In[61]:


tok_enum = list(enumerate(nltk_tokens))


# In[16]:


len(nltk_tokens)


# ## All the verb forms

# In[16]:


sortir_forms = set(('sortir sors sort sortons sortez sortent sorti '
                    'sortis sortais sortait sortiez sortiez sortaient '
                    'sortis sortit sortîtes sortîmes sortirent sortirai '
                    'sortiras sortira sortirons sortirez sortiront '
                    'sortirais sortirait sortirions sortiriez sortiraient '
                    'sorte sortes sorte sortions sortiez sortent sortisse sortisses '
                    'sortît sortissions sortissiez sortissent sortant').split())

partir_forms = set(('partir pars part partons partez partent parti partis partais '
                    'partait partions partiez partaient partîmes partirent '
                    'partirai partiras partira partirons partirez partiront '
                    'partirais partirait partirions partiriez partiraient '
                    'parte partes parte partions partiez partisse partisses '
                    'partît partissions partissiez partissent partant').split())

quitter_forms = set(('quitter quitte quittes quitte quittons quittez quittent quittais '
                     'quittais quittait quittai quittas quitta quittâmes quittâmes '
                     'quittâtes quittèrent quitté quitterai quitteras quittera '
                     'quitterons quitterez quitteron quitterais quitterais '
                     'quitterait quitterions quitteriez quitteraient quitte '
                     'quittes quittions quittiez quittent quittant quittasse '
                     'quittasses quittât quittassions quittassiez quittassent').split())

laisser_forms = set(('laisser laisse lasses laisse laissons laissez laissent laissais '
                     'laissé laissait laissions laissiez laissaient laissai laissas '
                     'laissa laissâmes laissâtes laissèresnt laisserai laisseras laissera '
                     'laisserons laisserez laisseront laisserais laisserais laisserait '
                     'laisserions laisseriez laisseraient laisse laisses laisse laissions '
                     'laissiez laissasse laissesses laissât laissassions laissassiez '
                     'laissassent laissant').split())


# In[98]:


def finder(nltk_tokens, verb_forms, n_words):
    """ pass in all the verb forms and return the context surrounding each verb"""
    indexes = [i for i, word in enumerate(nltk_tokens) if word in verb_forms]
    results = []
    for i in indexes:
        context = (nltk_tokens[i - n_words:i] +         # words to the left
                   # ['__' + nltk_tokens[i] + '__'] +    # search word
                   [nltk_tokens[i]] +                    # search word
                   nltk_tokens[i + 1:i + n_words + 1])   # words to the right
        results.append(context)
    print('len of results', len(results))
    return results


# ## Find and store the four verbs and their contexts

# In[100]:


sortir_concordance = finder(nltk_tokens, sortir_forms, 5)


# In[101]:


partir_concordance = finder(nltk_tokens, partir_forms, 5)


# In[102]:


quitter_concordance = finder(nltk_tokens, quitter_forms, 5)


# In[103]:


laisser_concordance = finder(nltk_tokens, laisser_forms, 5)


# In[106]:


all_concordances = [sortir_concordance,
                    partir_concordance,
                    quitter_concordance,
                    laisser_concordance]


# ## Process the concordance results
#
# Modifying `spaCy` to tag for part of speech, but disabling `parser` and `ner`.

# In[113]:


import spacy

nlp = spacy.load('fr', disable=['parser', 'ner'])


# In[52]:


len(sortir_concordance)


# In[108]:


UNDESIRED = ['PUNCT', 'NUM']  # could add determiners 'DET' here


# In[118]:


def framer(concordance_list):
    """ takes the concordance list and
        returns a list of relevant pairs"""
    i = 0
    results = []
    for phrase in concordance_list:
        spacyd = nlp(' '.join(phrase))

        psd = [(token.text, token.pos_) for token in spacyd
               if token.pos_ not in UNDESIRED]
        results.append(psd)

    i += 1
    if i % 1000 == 0:
        print(i)

    return results


# In[119]:


tagged_concs = []
for i, c in enumerate(all_concordances):
    tagged_concs.append(framer(c))
    print('done with ', i)


# In[120]:


import pickle


# ### Pickle the processed results
#
# This notebook is getting really big, and slow to run. Pickling allows me to
# close kernal, and reload the variables that I want at this point.

# In[122]:


with open('tagged_verb_concs.pkl', 'wb') as f:
    pickle.dump(tagged_concs, f)


# ### Open the processed results

# In[3]:


import pandas as pd
import pickle  # reopen after closing the kernal


# In[4]:


with open('tagged_verb_concs.pkl', 'rb') as f:
    tagged_concs = pickle.load(f)


# ## Align, create, and display `FreqDists`

# In[47]:


from nltk.probability import FreqDist


# Note the format of `tagged_concs`:
#
# 0. sortir_concordance
#     - phrase
#         - `(word, POS)`
#         - `(word, POS)`
#         - `(word, POS)`
#     - phrase
#         - `(word, POS)`
#         - `(word, POS)`
#         - `(word, POS)`
# 1. partir_concordance
#     - phrase (etc.)
# 2. quitter_concordance
# 3. laisser_concordance

# In[160]:


sortir = tagged_concs[0]
partir = tagged_concs[1]
quitter = tagged_concs[2]
laisser = tagged_concs[3]


# In[228]:


def aligner(tagged_conc, verb_forms, display='pos'):
    """ Pass in `tagged_conc` and `verb_forms`
        process them and print the results.

        `display` can be set to `pos` or `word`
        default for `display` == 'pos'
        """

    aligned = []
    # iterate over phrases
    for phrase in tagged_conc:
        # iterate over items in phrase
        for i, (word, pos) in enumerate(phrase):
            # align words around node verb form
            if word in verb_forms and pos == 'VERB':
                context = {}
                # select everything to the left
                context['left'] = phrase[:i]
                # select the found form of the verb
                context['node'] = phrase[i]
                # select everything to the right
                context['right'] = phrase[i + 1:]
                aligned.append(context)

    """ Pass in aligned searches and display the context """

    # ************************************************************
    # Build the lists
    # ************************************************************

    # (l|r)_dist is the left or right distribution of words
    l_dist = {}
    # verb forms
    node = []
    # r_dist
    r_dist = {}

    # ************************************************************
    # Organize the results
    # ************************************************************

    # `display` from kwargs
    if display == 'pos':
        n = 1
    elif display == 'word':
        n = 0

    CARE_ABOUT = ['NOUN' 'ADJ', 'PROPN']

    for phrase in aligned:
        # left context `reversed` to work from right to left
        for i, (word, pos) in reversed(list(enumerate(phrase['left']))):
            choose = (word, pos)
            if pos in CARE_ABOUT:
                try:
                    l_dist[i] += [choose[n]]
                except KeyError:
                    l_dist[i] = []
                    l_dist[i] += [choose[n]]

        # save node words
        node += [(phrase['node'])]

        # right context
        for i, (word, pos) in enumerate(phrase['right']):
            choose = (word, pos)
            if pos in CARE_ABOUT:
                try:
                    r_dist[i] += [choose[n]]
                except KeyError:
                    r_dist[i] = []
                    r_dist[i] += [choose[n]]

    # ************************************************************
    # `FreqDist` of the left and right-hand contexts
    # ************************************************************

    dists = [l_dist, r_dist]

    # iterate over each r and l distribution
    which = ['l_dist', 'r_dist']

    # `ch` type(int) is which channel, in a given `dist` or distribution
    for ch, dist in enumerate(dists):
        # go over the distribution for each position in order
        total_dist = {}

        reverse = True  # control flow (big to small)
        # if the channel selected is `'r_dist'` then go small to big
        if which[ch] == 'r_dist':
            reverse = False

        # build `FreqDist` objects from the `value_lists`
        for index, value_list in sorted(dist.items(), reverse=reverse):
            # print(f'{which[ch]}-{key}')
            v_list = []
            # `dist[index]` gets the appropriate `value_list`
            for k, v in FreqDist(dist[index]).items():
                v_list.append((k, v))
                total_dist[f'{which[ch]}-{index}'] = v_list

    # ************************************************************
    # Display the results
    # ************************************************************

            print(f'{which[ch]}-{index}', sorted(v_list,
                                                 reverse=True,
                                                 key=lambda x: x[1])[:20])
        if reverse:
            print()
            print('NODE WORD:', sorted(FreqDist(node).items(),
                                       reverse=True,
                                       key=lambda x: x[1])[:10])
            print()
#     return total_dist


# In[229]:


aligner(sortir, sortir_forms, display='word')


# In[230]:


aligner(partir, partir_forms, display='word')


# In[232]:


aligner(laisser, laisser_forms, display='word')


# In[233]:


aligner(quitter, quitter_forms, display='word')


# # NER Analysis
#
# Frequency distribution of named entities for each domain in corpus

# In[259]:


from glob import glob
import justext
import spacy
import re
nlp = spacy.load('fr_core_news_sm')


# In[252]:


PAGE_GLOB = glob('/Users/Kyle/Documents/archives/Winter 2018/'
                 'corpus-linguistics/french-news-html/'
                 'goscraper/20mar-links/*.html')


# In[243]:


def get_text(page):
    """This function takes an html page from a glob, for exmaple,
       and reads it and uses the justext module remove all boilerplate"""

    # reads the file
    page_string = open(page, 'rb').read()
    # creates a justext object
    paragraphs = justext.justext(page_string,
                                 justext.get_stoplist("French"))
    pageText = ''
    # if not boilerplate, adds to `pageText`
    for p in paragraphs:
        if not p.is_boilerplate:
            pageText += p.text + ' '
    return pageText


# ## Link Index
#
# Create an index link to scrape number.

# In[244]:


index_file = open('golang-link-index.txt', 'r').readlines()

url_index = []
for line in index_file:
    divided = line.split()
    url = re.sub(',', '', divided[0])
    index = re.findall(r'\d{2}mar-links/(\d+\.html)', divided[1])[0]
    url_index.append((url, index))


# In[256]:


def get_url(page_number):
    """ take the indexed filename and return the actual link """
    url = [url for url, index in url_index if index == page_number]
    return url[0]


# In[246]:


def domain_analyze(url):
    """ determines what domain the url is from """

    re_domain = r'https?://((?:www\.)?.*?\..*?)/'
    domain = re.search(re_domain, url).group(1)

    return domain


# In[247]:


class ArticleData:
    """ A place to store all of the things we find in each article

        `nes` is 'named enetities' """
    def __init__(self, article_text, url, domain, entity_labels, nes):
        self.article_text = article_text
        self.url = url
        self.domain = domain
        self.entity_labels = entity_labels
        self.nes = nes


# In[260]:


# `PAGE_GLOB` is a list of scraped pages `1.html`, `2.html`, etc.
# `indexed_path_html` is a file path to the stored html on my computer
result_list = []

TOTAL = len(PAGE_GLOB)
i = 0
for indexed_path_html in PAGE_GLOB:
    # get text
    article_text = get_text(indexed_path_html)
    # get the page number out of the path
    page_number = re.findall(r'(\d+\.html)', indexed_path_html)[0]
    # get the url
    url = get_url(page_number)
    # get the domain
    domain = domain_analyze(url)

    # analyze the text
    doc = nlp(article_text)

    # store the labels
    labels = []
    nes = []

    # get the named entities
    for ent in doc.ents:
        labels.append(ent.label_)
        nes.append(ent.text)

    Results = ArticleData(article_text, url, domain, labels, nes)

    if i % 1000 == 0:
        print(i)
    i += 1

    result_list.append(Results)

# In[268]:


len(result_list)


# In[265]:


with open('french_corpus_info.pkl', 'wb') as f:
    pickle.dump(result_list, f)


# In[269]:


with open('french_corpus_info.pkl', 'rb') as fa:
    asdf = pickle.load(fa)


# In[270]:


len(asdf)  # proves that the pickle worked.


# In[273]:


# article_text, url, domain, entity_labels, nes

domain_distribution = {}

for Res in result_list:
    if Res.domain not in domain_distribution.keys():
        domain_distribution[Res.domain] = 1
    else:
        domain_distribution[Res.domain] += 1


# In[275]:


len(domain_distribution.keys())


# ## Most common named entities

# In[299]:


most_common_nes = {}
for domain in domain_distribution.keys():
    for Res in result_list:
        if Res.domain == domain:
            if domain not in most_common_nes.keys():
                most_common_nes[domain] = Res.nes
            else:
                most_common_nes[domain] += Res.nes

# take the 30 most common domains
most_common_domains = FreqDist(domain_distribution).most_common(30)

for domain, count in most_common_domains:
    for domain2, nes in most_common_nes.items():
        if domain == domain2:
            print(count, domain, FreqDist(nes).most_common(11))
            print()


# ## Most common categories of NEs

# In[291]:


most_common_nes = {}
for domain in domain_distribution.keys():
    for Res in result_list:
        if Res.domain == domain:
            if domain not in most_common_nes.keys():
                most_common_nes[domain] = Res.entity_labels
            else:
                most_common_nes[domain] += Res.entity_labels

most_common_domains = FreqDist(domain_distribution).most_common(30)

for domain, count in most_common_domains:
    for domain2, nes in most_common_nes.items():
        if domain == domain2:
            print(count, domain, FreqDist(nes).most_common(11))


# ## Average article length

# In[293]:


from statistics import mean


# In[298]:


most_common_nes = {}
for domain in domain_distribution.keys():
    for Res in result_list:
        if Res.domain == domain:
            if domain not in most_common_nes.keys():
                most_common_nes[domain] = [len(Res.article_text)]
            else:
                most_common_nes[domain] += [len(Res.article_text)]

most_common_domains = FreqDist(domain_distribution).most_common(30)

for domain, count in most_common_domains:
    for domain2, lengths in most_common_nes.items():
        if domain == domain2:
            print('{:5} {:<25} {:<20}'.format(count, domain, mean(lengths)))


# ## Average TTR

# In[300]:


from nltk.tokenize import word_tokenize


# In[306]:


def ttr(article_text):
    """ Returns the nltk-word_tokenized type to token ratio of the article """
    tokens = word_tokenize(article_text)
    types = set(tokens)

    try:
        ttr = len(types) / len(tokens)
    except ZeroDivisionError:
        ttr = 0

    return ttr


# In[307]:


most_common_dict = {}
for domain in domain_distribution.keys():
    for Res in result_list:
        if Res.domain == domain:
            if domain not in most_common_dict.keys():
                most_common_dict[domain] = [ttr(Res.article_text)]
            else:
                most_common_dict[domain] += [ttr(Res.article_text)]

most_common_domains = FreqDist(domain_distribution).most_common(30)

for domain, count in most_common_domains:
    for domain2, lengths in most_common_dict.items():
        if domain == domain2:
            print('{:5} {:<25} {:<20}'.format(count, domain, mean(lengths)))
