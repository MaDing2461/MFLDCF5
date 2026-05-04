__all__ = [
    'ForensicHubSegformerB3',
    'ForensicHubIMLViT',
    'ForensicHubMesorch',
    'Segformerb3',
]


def __getattr__(name):
    if name in ('ForensicHubSegformerB3', 'Segformerb3'):
        from .segformer import ForensicHubSegformerB3, Segformerb3
        values = {
            'ForensicHubSegformerB3': ForensicHubSegformerB3,
            'Segformerb3': Segformerb3,
        }
        return values[name]

    if name == 'ForensicHubIMLViT':
        from .iml_vit import ForensicHubIMLViT
        return ForensicHubIMLViT

    if name == 'ForensicHubMesorch':
        from .mesorch import ForensicHubMesorch
        return ForensicHubMesorch

    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
