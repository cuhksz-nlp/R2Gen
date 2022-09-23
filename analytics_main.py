# exp setup
import time

from _global import timer
from _global.argument_parser import ArgumentParser
from data_processors.data_processor import DataProcessor
from data_processors.negation_detection import NegationDetection
from modules.tokenizers import Tokenizer


def main():
    start_time = time.time()
    # parse arguments
    args = ArgumentParser().args

    # r2gen split
    args.is_new_random_split = 0

    # associate_iu_r2gen_kaggle_by_id(args)
    # mip_j_split = json.loads(open(args.iu_mesh_impression_path, 'r').read())
    # mip_j = {sid: sample for split, split_sample in mip_j_split.items() for sid, sample in split_sample.items()}
    # json.dump(mip_j, open(args.iu_mesh_impression_path.replace("_split", ""), 'w'))

    data_processor = DataProcessor(args)
    # print("Is association file valid: ", data_processor.validate_association())
    tokenizer = Tokenizer(args, data_processor)
    negation_detection = NegationDetection(args)
    for split, split_sample in data_processor.iu_mesh_impression_split.items():
        for kid, sample in split_sample.items():
            negation_detection.get_lemmatize_doc_object(sample)
    # exp_stats = ExperimentsStatistics(tokenizer, args.exp)
    # print("exp: ", args.exp, " ", exp_stats.stats)
    #
    # print("######### before split#########")
    # data_processor.analyze.print_normal_percentage()
    # data_processor.analyze.print_no_index_percentage()
    # data_processor.analyze.print_empty_mesh_asc_percentage()
    # print("Is association file valid: ", data_processor.validate_association())
    #
    # plot = Plot(args, data_processor.analyze)
    # # normal ratio
    # plot.plot_stacked_bar(num=1, xs=["t_ratio", "normal_ratio"], ys=["split"],
    #                       colors=[plot.abnormal_color, plot.normal_color], labels=['Abnormal', 'Normal'],
    #                       number_of_col_in_legend=2, plot_name="[R2gen Split]Normal to Abnormal ratio")
    # # indexed ratio
    # plot.plot_stacked_bar(num=2, xs=["t_ratio", "no_index_ratio"], ys=["split"],
    #                       colors=[plot.indexed_color, plot.no_index_color], labels=['Indexed', 'No Indexing'],
    #                       number_of_col_in_legend=2, plot_name="[R2gen Split]Indexed to No Indexing ratio")
    # # empty mesh ratio
    # plot.plot_stacked_bar(num=3, xs=["t_ratio", "no_mesh_ratio"], ys=["split"],
    #                       colors=[plot.mesh_color, plot.no_mesh_color], labels=['MeSH', 'No MeSH'],
    #                       number_of_col_in_legend=2, plot_name="[R2gen Split]Mesh to No MeSH ratio")
    #
    # # new split
    # args.is_new_random_split = 1
    # data_processor = DataProcessor(args)
    # print("######### after split#########")
    # data_processor.analyze.print_normal_percentage()
    # data_processor.analyze.print_no_index_percentage()
    # data_processor.analyze.print_empty_mesh_asc_percentage()
    # print("Is association file valid: ", data_processor.validate_association())
    #
    # plot = Plot(args, data_processor.analyze)
    # # normal ratio
    # plot.plot_stacked_bar(num=4, xs=["t_ratio", "normal_ratio"], ys=["split"],
    #                       colors=[plot.abnormal_color, plot.normal_color], labels=['Abnormal', 'Normal'],
    #                       number_of_col_in_legend=2, plot_name="[New Split]Normal to Abnormal ratio")
    # # indexed ratio
    # plot.plot_stacked_bar(num=5, xs=["t_ratio", "no_index_ratio"], ys=["split"],
    #                       colors=[plot.indexed_color, plot.no_index_color], labels=['Indexed', 'No Indexing'],
    #                       number_of_col_in_legend=2, plot_name="[New Split]Indexed to No Indexing ratio")
    # # empty mesh ratio
    # plot.plot_stacked_bar(num=6, xs=["t_ratio", "no_mesh_ratio"], ys=["split"],
    #                       colors=[plot.mesh_color, plot.no_mesh_color], labels=['MeSH', 'No MeSH'],
    #                       number_of_col_in_legend=2, plot_name="[New Split]MeSH to No MeSH ratio")
    # plt.show()
    timer.time_executed(start_time, "R2Gen.Analysis")


# def associate_iu_r2gen_kaggle_by_id(args):
#     kaggle_iu_reports = csv.reader(open(args.kaggle_iu_reports_path, 'r'))
#     r2gen_ann = json.loads(open(args.ann_path, 'r').read())
#     next(kaggle_iu_reports)
#     r2gen_splits_ids_reports = {
#         split: [{sample["id"]: sample["report"]} for sample in samples]
#         for split, samples in r2gen_ann.items()
#     }
#
#     kaggle_uids_mesh_impression = {
#         line[0]: {"MeSH": line[1], "report": line[6], "impression": line[7]}
#         for line in kaggle_iu_reports
#     }
#
#     unmatched = dict(train=[], val=[], test=[])
#     matched = dict(train={}, val={}, test={})
#     for split, samples in r2gen_splits_ids_reports.items():
#         for sample in samples:
#             for r2gen_id, r2gen_report in sample.items():
#                 uid = r2gen_id.split('_')[0].replace("CXR", "")
#                 if uid in kaggle_uids_mesh_impression:
#                     kaggle_report = kaggle_uids_mesh_impression[uid]["report"]
#                     if r2gen_report == kaggle_report:
#                         iu_mesh = kaggle_uids_mesh_impression[uid]["MeSH"]
#                         mesh_text = ""
#                         attr_text = ""
#                         mesh_attr_text = ""
#                         for mesh_info in iu_mesh.split(';'):
#                             if '/' in mesh_info:
#                                 mesh_attr = mesh_info.split('/')
#                                 ma_text = ""
#                                 if ',' in mesh_attr[0]:
#                                     for ma in mesh_attr[0].split(','):
#                                         ma_text += " <mesh:{}>".format(ma.strip().replace(' ', '_'))
#                                 else:
#                                     ma_text = " <mesh:{}>".format(mesh_attr[0].strip().replace(' ', '_'))
#                                 mesh_text += ma_text
#                                 attr_text += " <attr:{}>".format(mesh_attr[1].strip().replace(' ', '_'))
#                                 mesh_attr_text += "{} <attr:{}>".format(ma_text,
#                                                                         mesh_attr[1].strip().replace(' ', '_'))
#                         matched[split][r2gen_id] = {
#                             "iu_mesh": iu_mesh, "mesh": mesh_text, "attr": attr_text,
#                             "mesh_attr": mesh_attr_text,
#                             "impression": kaggle_uids_mesh_impression[uid]["impression"]}
#                         # matched[split].append(matched_info)
#                     else:
#                         unmatched[split][r2gen_id] = {"r2gen_uid": uid, "r2gen_report": r2gen_report,
#                                                       "kaggle_report": kaggle_report}
#                         # unmatched[split].append(unmatched_info)
#                 else:
#                     unmatched_info = {
#                         r2gen_id: {"r2gen_uid": uid, "r2gen_report": r2gen_report, "kaggle_report": ""}}
#                     unmatched[split].append(unmatched_info)
#     if not os.path.exists(args.iu_mesh_impression_path):
#         os.mknod(args.iu_mesh_impression_path)
#     json.dump(matched, open(args.iu_mesh_impression_path, 'w'))
#     return matched

if __name__ == '__main__':
    main()
#############################################################################################
