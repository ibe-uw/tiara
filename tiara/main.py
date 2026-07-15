import argparse
import sys
from collections import Counter
import gzip


minimum_sequence_length = 3000
default_prob_cutoff = [0.65, 0.65]
short_mapping = {
    "mit": "mitochondrion",
    "pla": "plastid",
    "bac": "bacteria",
    "arc": "archaea",
    "euk": "eukarya",
    "unk": "unknown",
    "pro": "prokarya",
}

first_nnet_kmer_to_params = {
    4: dict(
        fname="first_nnet_kmer_4.pkl", k=4, hidden_1=2048, hidden_2=2048, dropout=0.2
    ),
    5: dict(
        fname="first_nnet_kmer_5.pkl", k=5, hidden_1=2048, hidden_2=2048, dropout=0.2
    ),
    6: dict(
        fname="first_nnet_kmer_6.pkl", k=6, hidden_1=2048, hidden_2=1024, dropout=0.2
    ),
}
second_nnet_kmer_to_params = {
    4: dict(
        fname="second_nnet_kmer_4.pkl", k=4, hidden_1=256, hidden_2=128, dropout=0.2
    ),
    5: dict(
        fname="second_nnet_kmer_5.pkl", k=5, hidden_1=256, hidden_2=128, dropout=0.2
    ),
    6: dict(
        fname="second_nnet_kmer_6.pkl", k=6, hidden_1=256, hidden_2=128, dropout=0.5
    ),
    7: dict(
        fname="second_nnet_kmer_7.pkl", k=7, hidden_1=128, hidden_2=64, dropout=0.2
    ),
}

classes_list = ["org", "bac", "arc", "euk", "unk1", "pla", "unk2", "mit"]


description = """tiara - a deep-learning-based approach for identification of eukaryotic sequences 
in the metagenomic data powered by PyTorch.  

The sequences are classified in two stages:

- In the first stage, the sequences are classified to classes: 
      archaea, bacteria, prokarya, eukarya, organelle and unknown.
- In the second stage, the sequences labeled as organelle in the first stage 
      are classified to either mitochondria, plastid or unknown.
"""


def main(test=None):
    if test:
        perform_test()
    else:
        args = parse_arguments()
        import pkg_resources
        import os
        import time

        import torch

        from tiara.src.classification import Classification
        from tiara.src.utilities import sort_type, write_to_fasta

        torch.set_num_threads(args.threads)

        first_stage_model_params = first_nnet_kmer_to_params[args.first_stage_kmer]
        second_stage_model_params = second_nnet_kmer_to_params[args.second_stage_kmer]

        first_stage_tfidf_fname = f"k{first_stage_model_params['k']}-first-stage"
        second_stage_tfidf_fname = f"k{second_stage_model_params['k']}-second-stage"

        if len(args.prob_cutoff) == 1:
            prob_cutoff_1, prob_cutoff_2 = args.prob_cutoff[0], args.prob_cutoff[0]
        else:
            prob_cutoff_1, prob_cutoff_2 = args.prob_cutoff

        first_stage_model_params["prob_cutoff"] = prob_cutoff_1
        second_stage_model_params["prob_cutoff"] = prob_cutoff_2
        first_stage_model_params["fragment_len"] = 5000
        second_stage_model_params["fragment_len"] = 5000
        first_stage_model_params["dim_out"] = 5
        second_stage_model_params["dim_out"] = 3

        params = [first_stage_model_params, second_stage_model_params]
        nnet_weights = [
            pkg_resources.resource_filename(__name__, "models/nnet-models/" + path)
            for path in [
                first_stage_model_params["fname"],
                second_stage_model_params["fname"],
            ]
        ]
        tfidfs = [
            pkg_resources.resource_filename(__name__, "models/tfidf-models/" + path)
            for path in [first_stage_tfidf_fname, second_stage_tfidf_fname]
        ]

        classifier = Classification(
            min_len=args.min_len,
            nnet_weights=nnet_weights,
            params=params,
            tfidf=tfidfs,
            threads=args.threads,
            quality=args.quality
        )
        start_time = time.time()
        results = classifier.classify(args.input, args.verbose)
        tot_time = time.time() - start_time
        print(f"Classification took {tot_time} seconds.")
        print(f"{len(results)/tot_time} sequences per second.")
        print(f"{sum(len(x.seq) for x in results) / tot_time} base pairs per second.")
        log = prepare_statistics(results)
        output = prepare_output(results, args.probabilities)
        log += "\n\n\nModels used:\n\tTf-idf:\n"
        log += "".join(f"\t\t{model_path}\n" for model_path in tfidfs)
        log += "\tNeural net weights:\n"
        log += "".join(f"\t\t{nnet_weight}\n" for nnet_weight in nnet_weights)
        log += f"\tHyperparameters:\n"
        for i, param in enumerate(params):
            log += f"\t\tStage {i}:\n"
            for param, value in param.items():
                log += f"\t\t\t{param}: {value}\n"
        print("Classification done.")
        print(prepare_statistics(results))
        if args.output:
            directory, fname = os.path.split(args.output)
            if not os.path.exists(directory) and directory:
                os.makedirs(directory)
            if args.gzip:
                with gzip.open(args.output + ".gz", "wt") as target:
                    target.write(output)
                print(f"Output saved to {args.output}.gz.")
                with gzip.open(
                    os.path.join(directory, "log_" + fname) + ".gz", "wt"
                ) as target:
                    target.write(log)
                print(
                    f"Log file saved to {os.path.join(directory, 'log_' + fname)}.gz."
                )
            else:
                with open(args.output, "w") as target:
                    target.write(output)
                print(f"Output saved to {args.output}.")
                with open(os.path.join(directory, "log_" + fname), "w") as target:
                    target.write(log)
                print(f"Log file saved to {os.path.join(directory, 'log_' + fname)}.")
        else:
            print(output)
        if args.to_fasta:
            if args.output:
                directory, _ = os.path.split(args.output)
            else:
                directory = "."
            grouped = sort_type(results)
            _, name = os.path.split(args.input)
            if "all" in args.to_fasta:
                classes = list(short_mapping.keys())
            else:
                classes = args.to_fasta
            for cls in classes:
                if grouped[short_mapping[cls]]:
                    fname = short_mapping[cls] + "_" + name
                    if args.gzip:
                        with gzip.open(
                            os.path.join(directory, fname) + ".gz", "wt"
                        ) as handle:
                            write_to_fasta(handle, grouped[short_mapping[cls]])
                    else:
                        with open(os.path.join(directory, fname), "w") as handle:
                            write_to_fasta(handle, grouped[short_mapping[cls]])
        print()


def prepare_output(results, probs=False):
    output = "sequence_id\tclass_fst_stage\tclass_snd_stage"
    if probs:
        output += "\t" + "\t".join(classes_list)
    output += "\n"
    for record in results:
        output += record.generate_line(probs) + "\n"
    return output


def prepare_statistics(results):
    fst_iteration_statistics = Counter(x.cls[0] for x in results)
    snd_iteration_statistics = Counter(x.cls[1] for x in results)
    log = "First iteration statistics:\n"
    log += "".join(
        f"\t{_cls}: {count}\n"
        for _cls, count in sorted(fst_iteration_statistics.items(), key=lambda x: x[0])
    )
    if not all(x == "n/a" for x in snd_iteration_statistics.keys()):
        log += "Second iteration statistics:\n"
        log += "".join(
            f"\t{_cls}: {count}\n"
            for _cls, count in sorted(
                snd_iteration_statistics.items(), key=lambda x: x[0]
            )
            if _cls != "n/a"
        )
    return log


def perform_test():
    import pkg_resources
    import time

    import torch

    from tiara.src.classification import Classification

    nnet_weights = [
        pkg_resources.resource_filename(__name__, "models/nnet-models/" + path)
        for path in [
            "first_nnet_kmer_6.pkl",
            "second_nnet_kmer_7.pkl",
        ]
    ]
    tfidfs = [
        pkg_resources.resource_filename(__name__, "models/tfidf-models/" + path)
        for path in [
            "k6-first-stage",
            "k7-second-stage",
        ]
    ]
    torch.set_num_threads(1)
    fragment_len = 5000
    params = [
        first_nnet_kmer_to_params[6],
        second_nnet_kmer_to_params[7],
    ]
    params[0]["prob_cutoff"] = 0.65
    params[1]["prob_cutoff"] = 0.65
    params[0]["fragment_len"] = 5000
    params[1]["fragment_len"] = 5000
    params[0]["dim_out"] = 5
    params[1]["dim_out"] = 3
    data_template = "test/test_data/"
    classifier = Classification(
        min_len=fragment_len,
        nnet_weights=nnet_weights,
        params=params,
        tfidf=tfidfs,
    )
    tests_passed = 0
    for i, source_dataset in enumerate(
        ["archaea", "bacteria", "eukarya", "mitochondria", "plast"]
    ):
        print(f"Testing file {i + 1}/5: {source_dataset + '_fr.fasta.gz'}")
        fasta_fpath = pkg_resources.resource_filename(
            __name__, data_template + source_dataset + "_fr.fasta.gz"
        )
        target_output_fpath = pkg_resources.resource_filename(
            __name__, data_template + source_dataset + "_out.txt"
        )
        start_time = time.time()
        results = classifier.classify(fasta_fpath, verbose=True)
        end_time = time.time()
        tot_time = end_time - start_time
        print(f"Testing took {tot_time} seconds.")
        print(f"{len(results) / tot_time} sequences per second.")
        print(f"{sum(len(x.seq) for x in results) / tot_time} base pairs per second.")
        log = prepare_statistics(results)
        print("Classification statistics:")
        print(log)
        target_results = {}
        with open(target_output_fpath, "r") as handle:
            next(handle)
            for line in handle:
                desc, cls1, cls2 = line.strip().split("\t")
                target_results[desc] = (cls1, cls2)
        test_results = {
            record.desc: (record.cls[0], record.cls[1]) for record in results
        }
        if all(
            target_results[key] == test_results[key] for key in target_results.keys()
        ):
            print("Test passed.")
            tests_passed += 1
        else:
            print("Test failed.")
        print()
    print(f"Passed {tests_passed}/5 tests.")
    if tests_passed == 5:
        print("Tests successful.")
    else:
        print("Tests failed.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-i",
        "--input",
        metavar="input",
        help="A path to a fasta file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="output",
        help="A path to output file. If not provided, the result is printed to stdout.",
        default=None,
    )
    parser.add_argument(
        "-m",
        "--min_len",
        help=f"""Minimum length of a sequence. Sequences shorter than min_len are discarded. 
        Default: {minimum_sequence_length}.""",
        type=int,
        default=minimum_sequence_length,
    )
    parser.add_argument(
        "--first_stage_kmer",
        "--k1",
        help=f"k-mer length used in the first stage of classification. Default: 6.",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--second_stage_kmer",
        "--k2",
        help=f"k-mer length used in the second stage of classification. Default: 7.",
        type=int,
        default=7,
    )
    parser.add_argument(
        "-p",
        "--prob_cutoff",
        metavar="cutoff",
        help=f"""Probability threshold needed for classification to a class. 
        If two floats are provided, the first is used in a first stage, the second in the second stage
        Default: {default_prob_cutoff}.""",
        nargs="+",
        type=float,
        default=default_prob_cutoff,
    )

    parser.add_argument(
        "--to_fasta",
        "--tf",
        metavar="class",
        help="""Write sequences to fasta files specified in the arguments to this option.
        The arguments are: mit - mitochondria, pla - plastid, bac - bacteria, 
        arc - archaea, euk - eukarya, unk - unknown, pro - prokarya, 
        all - all classes present in input fasta (to separate fasta files).""",
        nargs="+",
        type=str,
        default=[],
    )

    parser.add_argument(
        "-t", "--threads", help="Number of threads used.", type=int, default=1
    )

    parser.add_argument(
        "--probabilities",
        "--pr",
        action="store_true",
        help="""Whether to write probabilities of individual classes for each sequence to the output.""",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="""Whether to display some additional messages and progress bar during classification.""",
    )

    parser.add_argument(
        "--gzip",
        "--gz",
        action="store_true",
        help="""Whether to gzip results or not.""",
    )

    parser.add_argument(
        "-q",
        "--quality",
        action="store_true",
        help="""Wheter to export the input fastq quality value to the output""",
    )

    return parser.parse_args(args=None if sys.argv[1:] else ["--help"])
