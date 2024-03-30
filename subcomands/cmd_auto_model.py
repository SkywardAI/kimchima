from pkg.auto import Auto


class CommandAuto:

    @staticmethod
    def auto(args):
        model = Auto(model_name_or_path=args.model_name_or_path)
        embeddings = model.get_embeddings(text='Melbourne')
        print(embeddings)
        embeddings = model.get_embeddings(text=['Melbourne', 'Sydney'])
        print(embeddings)
        return embeddings