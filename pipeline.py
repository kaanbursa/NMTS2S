class Vocabulary():
    def __init__(self, token_to_index=None):

        if token_to_index is None:
            token_to_index = {}
        self._token_to_index = token_to_index
        self._idx_to_token = {k: v for v, k in self._token_to_index.items()}


    def to_serializable(self):
        """ Returns a dictionary that can be serialized"""
        return {'token_to_idx' : self._token_to_index}

    @classmethod
    def from_serializable(cls, contents):
        """Instaites the Vocabulary from a serialized dictionary"""
        retun cls(**contents)


    def add_token(self, token):
        """
        :param token: the item thats gonna be added to Vocabulary
        :return: the integer corresponding
        """
        if token not in self._token_to_index:
            index = self._token_to_index[token]
        else:
            index = len(self._token_to_index)
            self._token_to_index[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """Add a list of tokens
        :param Tokens (list)
        :return list of indices
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self,token):
        """ Retrieve index of token
        :param Token(str) the desired token
        :return Index(int): index corresponding
        """

        return self._token_to_index[token]

    def lookup_index(self,index):
        if index not in self._idx_to_token:
            raise KeyError('the index (%d) is not in the Vocabulary' % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_index)


def SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_index = None, unk_token="<UNK>", mask_token="<MASK>", begin_seq_token="<BEGIN>", end_seq_token="<END>"):
        super(Vocabulary, self).__init__(token_to_index)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_token = self.add_token(self._unk_token)
        self.begin_seq_token = self.add_token(self._begin_seq_token)
        self.end_seq_token = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token':self._unk_token, 'mask_token':self._mask_token, 'begin_seq_token': self._begin_seq_token, 'end_seq_token':self._end_seq_token})
        return contents

    def lookup_token(self, token):
        if self._unk_index >= 0:
            return self._token_to_idx.get(token, self._unk_index)
        else:
            return self._token_to_idx[token]
        


class NMTVectorizer(object):

    def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):

        """
        Args:
            source vocabulary: maps source words to integers
            target vocabulart: maps target words to integers
            max_source_length: longest sequence in the source dataset
            max_target_length: longest sequence in target dataset
        """
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    @classmethod
    def from_dataframe(cls, bitext_df):
        """
        Instantiate the vectorizer from the dataset dataframe

        Args:
            bitext_df: the parallel text dataset
        Returns:
            an instance of NMT vectorizer
        """
        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()
        max_source_length, max_target_length = 0, 0

        for _, row in bitext_df.iterrows():
            source_tokens = row["source_language"].split(" ")
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)

            target_tokens = row["target_language"].split(" ")
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                target_vocab.add_token(token)

        return cls(source_vocab, target_vocab, max_source_length, max_target_length)