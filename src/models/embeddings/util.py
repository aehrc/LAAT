from src.models.embeddings.embedding_layer import EmbeddingLayer


def init_embedding_layer(args, vocab):
    embedding_layer = EmbeddingLayer(embedding_mode=args.mode,
                                     embedding_size=args.embedding_size,
                                     pretrained_word_embeddings=vocab.word_embeddings,
                                     vocab_size=vocab.n_words())

    return embedding_layer
