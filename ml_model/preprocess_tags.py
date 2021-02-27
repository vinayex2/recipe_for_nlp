import spacy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helper Script for Label Studio commands')
    parser.add_argument('mode', choices=['init','refresh'], help='Mode for Initializing project, Refresh mode deletes current configurations')

    args = parser.parse_args()
    if args.mode =='init':
        init(os.getcwd())
    elif args.mode == 'refresh':
        refresh(os.getcwd())