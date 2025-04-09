import argparse
from train_options import TrainOptions

class CombinedOptions:
    """Class to handle options from both SeqNet and GAN"""
    def __init__(self):
        self.seqnet_parser = argparse.ArgumentParser(description="Train a combined model.")
        self.seqnet_parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file for SeqNet.")
        self.seqnet_parser.add_argument("--eval", action="store_true", help="Evaluate SeqNet performance.")
        self.seqnet_parser.add_argument("--resume", action="store_true", help="Resume from the specified checkpoint.")
        self.seqnet_parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
        self.seqnet_parser.add_argument("--use_seqnet", action="store_true", help="Use SeqNet model.")
        self.seqnet_parser.add_argument("--use_gan", action="store_true", help="Use GAN model.")
        self.seqnet_parser.add_argument("opts", nargs=argparse.REMAINDER, 
                                        help="Modify config options using the command-line")
        
        # GAN options will be parsed separately
        self.gan_options = None
        self.seqnet_args = None
        
    def parse(self):
        """Parse arguments"""
        # First parse basic args to determine which model(s) to use
        args, _ = self.seqnet_parser.parse_known_args()
        
        # Parse full arguments
        self.seqnet_args = self.seqnet_parser.parse_args()
        
        # If using GAN, parse its options
        if args.use_gan:
            self.gan_options = TrainOptions().parse(save_config=False)
            
        return self
