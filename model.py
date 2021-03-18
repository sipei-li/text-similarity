from models.Transformer import Transformer
from models.SiameseNet import SiameseNet

def create_models(args):
    print('Producing model...')
    if args.stucture == 'siamese':
        bert = Transformer(args)
        model = SiameseNet(args, bert)
    return model