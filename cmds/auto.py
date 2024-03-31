from pkg import Auto


class CommandAuto:

    @classmethod
    def auto(args):
        model = Auto(model_name_or_path=args.model_name_or_path)
        embeddings = model.get_embeddings(args.text)
        return embeddings

