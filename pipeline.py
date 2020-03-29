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

    def _vectorize(self, indices, vector_length = -1, mask_index=0):
        if vector_length < 0:
            vector_length = len(indices)
        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index
        return vector


    def _get_source_indices(self, text):
        """Return the vectorized source text
            Args:
                text (str):  the source
            Returns:
                indices (list): list of integers
        """
        indices = [self.source_vocab.begin_seq_index]
        indices.extend(self.source_vocab.look_token(token) for token in text.split(' '))
        indices.append(self.source_vocab.end_seq_index)
        return indices

    def _get_target_indices(self, text):
        indices = [self.target_vocab.lookup_token(token) for token in text.split(' ')]
        x_indices = [self.target_vocab.begin_seq_index] + indices
        y_indices = indices + [self.target_vocab.end_seq_index]
        return x_indices, y_indices


    def vectorize(self, source_text, target_text, use_dataset_max_lengths = True):
        """Retrun vectorizeded source and target text"""
        source_vector_length = -1
        target_vector_length = -1

        if use_dataset_max_lengths:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1

        source_indices = self._get_source_indices(source_text)
        source_vector = self_vectorize(source_indices, vector_length = source_vector_length, mask_index = self.source_vocab.mask_index)

        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(target_x_indices, vector_length = target_vector_length,
                                          mask_index = self.target_vocab.mask_index)
        target_y_vector = self._vectorize(target_y_indices, vector_length=target_vector_length,
                                          mask_index=self.target_vocab.mask_index)

        return {"source_vector": source_vector, "target_x_vector":target_x_vector,
                "target_y_vector":target_y_vector, "source_length":len(source_indices)}


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

    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(contents['source_vocab'])
        target_vocab = SequenceVocabulary.from_serializable(contents['target_vocab'])
        return cls(source_vocab=source_vocab,
                   target_vocab=target_vocab,
                   max_source_length=contents["max_source_length"],
                   max_target_length=contents['max_target_length'])

    def to_serializable(self):
        return {"source_vocab": self.source_vocab.to_serializable(),
                "target_vocab": self.target_vocab.to_serializable(),
                "max_source_length": self.max_source_length,
                "max_target_length": self.max_target_length}

