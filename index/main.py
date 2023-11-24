import os

import pandas as pd

from index import evaluation
from index.conf import PD_CDM_SRC, PPMI_DICT_SRC, LUXPARK_DICT_SRC, BIOFIND_DICT_SRC, AD_CDM_SRC
from index.embedding import GPT4Adapter
from index.evaluation import match_closest_descriptions, MatchingMethod, enrichment_analysis
from index.mapping import MappingTable
from index.parsing import MappingSource, DataDictionarySource
from dotenv import load_dotenv

from index.visualisation import scatter_plot_two_distributions, enrichment_plot, scatter_plot_all_cohorts

EVAL_PD = True
EVAL_AD = True

load_dotenv()
gpt4 = GPT4Adapter(api_key=os.getenv('GPT_KEY'))


def evaluate(datasets, labels, store_results=False, results_root_dir="resources/results/pd"):
    data_gpt = {}
    data_fuzzy = {}
    for idx, source in enumerate(datasets):
        acc_gpt = []
        acc_fuzzy = []
        for idy, target in enumerate(datasets):
            map_gpt = match_closest_descriptions(source, target)
            map_fuzzy = match_closest_descriptions(source, target, matching_method=MatchingMethod.FUZZY_STRING_MATCHING)
            if target == "jadni":
                print("check")
            if store_results:
                map_gpt.to_excel(results_root_dir + "/gpt_" + f"{labels[idx]}_to_{labels[idy]}.xlsx")
                map_fuzzy.to_excel(results_root_dir + "/fuzzy_" + f"{labels[idx]}_to_{labels[idy]}.xlsx")
            acc_gpt.append(round(evaluation.score_mappings(map_gpt), 2))
            acc_fuzzy.append(round(evaluation.score_mappings(map_fuzzy), 2))
        data_gpt[labels[idx]] = acc_gpt
        data_fuzzy[labels[idx]] = acc_fuzzy
    # transpose to have from -> to | row -> column like in the paper
    gpt = pd.DataFrame(data_gpt, index=labels).T
    fuzzy = pd.DataFrame(data_fuzzy, index=labels).T
    return gpt, fuzzy


# PD Mappings

if EVAL_PD:
    cdm_pd = MappingTable(MappingSource(PD_CDM_SRC, "Feature", "CURIE"))
    cdm_pd.joined_mapping_table["identifier"].to_csv("resources/cdm_curie.csv", index=False)
    cdm_pd.add_descriptions(DataDictionarySource(PD_CDM_SRC, "Feature", "Definition"))
    cdm_pd.compute_embeddings(gpt4)

    ppmi = MappingTable(MappingSource(PD_CDM_SRC, "PPMI", "CURIE"))
    ppmi.add_descriptions(DataDictionarySource(PPMI_DICT_SRC, "ITM_NAME", "DSCR"))
    ppmi.compute_embeddings(gpt4)

    luxpark = MappingTable(MappingSource(PD_CDM_SRC, "LuxPARK", "CURIE"))
    luxpark.add_descriptions(DataDictionarySource(LUXPARK_DICT_SRC, "Variable / Field Name", "Field Label"))
    luxpark.compute_embeddings(gpt4)

    biofind = MappingTable(MappingSource(PD_CDM_SRC, "BIOFIND", "CURIE"))
    biofind.add_descriptions(DataDictionarySource(BIOFIND_DICT_SRC, "ITM_NAME", "DSCR"))
    biofind.compute_embeddings(gpt4)

    lrrk2 = MappingTable(MappingSource(PD_CDM_SRC, "LRRK2", "CURIE"))
    lrrk2.add_descriptions(DataDictionarySource("resources/dictionaries/pd/LRRK2.xlsx", "Variable", "Label"))
    lrrk2.compute_embeddings(gpt4)

    opdc = MappingTable(MappingSource(PD_CDM_SRC, "OPDC", "CURIE"))
    opdc.add_descriptions(
        DataDictionarySource("resources/dictionaries/pd/OPDC.csv", "Variable Name", "Variable description"))
    opdc.compute_embeddings(gpt4)

    tpd = MappingTable(MappingSource(PD_CDM_SRC, "TPD", "CURIE"))
    tpd.add_descriptions(
        DataDictionarySource("resources/dictionaries/pd/TPD.csv", "Variable Name", "Variable description"))
    tpd.compute_embeddings(gpt4)

    pd_datesets = [opdc, tpd, biofind, lrrk2, luxpark, ppmi, cdm_pd]
    pd_datasets_labels = ["OPDC", "TPD", "Biofind", "LRRK2", "LuxPARK", "PPMI", "PASSIONATE"]

    # enrichment analysis
    luxpark_passionate_enrichment_gpt = enrichment_analysis(luxpark, cdm_pd, 20,
                                                            MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE)
    luxpark_passionate_enrichment_fuzzy = enrichment_analysis(luxpark, cdm_pd, 20, MatchingMethod.FUZZY_STRING_MATCHING)
    label1 = "Enrichment Plot LuxPARK to CDM"
    ppmi_passionate_enrichment_gpt = enrichment_analysis(ppmi, cdm_pd, 20, MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE)
    ppmi_passionate_enrichment_fuzzy = enrichment_analysis(ppmi, cdm_pd, 20, MatchingMethod.FUZZY_STRING_MATCHING)
    label2 = "Enrichment Plot PPMI to CDM"
    enrichment_plot(luxpark_passionate_enrichment_gpt, luxpark_passionate_enrichment_fuzzy, label1, save_plot=True)
    enrichment_plot(ppmi_passionate_enrichment_gpt, ppmi_passionate_enrichment_fuzzy, label2, save_plot=True)
    print(luxpark_passionate_enrichment_gpt)
    print(luxpark_passionate_enrichment_fuzzy)
    print(ppmi_passionate_enrichment_gpt)
    print(ppmi_passionate_enrichment_fuzzy)

    gpt_table, fuzzy_table = evaluate(pd_datesets, pd_datasets_labels)
    print("PD RESULTS:")
    print(gpt_table)
    print("-----------")
    print(fuzzy_table)
    print("-----------")

# AD Mappings

if EVAL_AD:
    cdm_ad = cdm_pd = MappingTable(MappingSource(AD_CDM_SRC, "Feature", "CURIE"))
    cdm_ad.add_descriptions(DataDictionarySource(PD_CDM_SRC, "Feature", "Definition"))
    cdm_ad.compute_embeddings(gpt4)

    a4 = MappingTable(MappingSource(AD_CDM_SRC, "A4", "CURIE"))
    a4.add_descriptions(DataDictionarySource("resources/dictionaries/ad/a4.csv", "FLDNAME", "TEXT"))
    a4.compute_embeddings(gpt4)

    abvib = MappingTable(MappingSource(AD_CDM_SRC, "ABVIB", "CURIE"))
    abvib.add_descriptions(DataDictionarySource("resources/dictionaries/ad/abvib.csv", "variable_name", "description"))
    abvib.compute_embeddings(gpt4)

    adni = MappingTable(MappingSource(AD_CDM_SRC, "ADNI", "CURIE"))
    adni.add_descriptions(DataDictionarySource("resources/dictionaries/ad/adni.csv", "FLDNAME", "TEXT"))
    adni.compute_embeddings(gpt4)

    aibl = MappingTable(MappingSource(AD_CDM_SRC, "AIBL", "CURIE"))
    aibl.add_descriptions(DataDictionarySource("resources/dictionaries/ad/aibl.csv", "Name", "Description"))
    aibl.compute_embeddings(gpt4)

    arwibo = MappingTable(MappingSource(AD_CDM_SRC, "ARWIBO", "CURIE"))
    arwibo.add_descriptions(
        DataDictionarySource("resources/dictionaries/ad/arwibo.csv", "Variable_Name", "Element_description"))
    arwibo.compute_embeddings(gpt4)

    dod_adni = MappingTable(MappingSource(AD_CDM_SRC, "DOD-ADNI", "CURIE"))
    # TODO most descriptions missing
    dod_adni.add_descriptions(DataDictionarySource("resources/dictionaries/ad/dod-adni.csv", "FLDNAME", "TEXT"))
    dod_adni.compute_embeddings(gpt4)

    edsd = MappingTable(MappingSource(AD_CDM_SRC, "EDSD", "CURIE"))
    edsd.add_descriptions(
        DataDictionarySource("resources/dictionaries/ad/edsd.xlsx", "Variable_Name", "Element_description"))
    edsd.compute_embeddings(gpt4)

    emif = MappingTable(MappingSource(AD_CDM_SRC, "EMIF", "CURIE"))
    emif.add_descriptions(DataDictionarySource("resources/dictionaries/ad/emif.xlsx", "Variable", "Description"))
    emif.compute_embeddings(gpt4)

    i_adni = MappingTable(MappingSource(AD_CDM_SRC, "I-ADNI", "CURIE"))
    # TODO about half of descriptions missing
    i_adni.add_descriptions(DataDictionarySource("resources/dictionaries/ad/i-adni.csv", "acronym", "variable"))
    i_adni.compute_embeddings(gpt4)

    jadni = MappingTable(MappingSource(AD_CDM_SRC, "JADNI", "CURIE"))
    jadni.add_descriptions(DataDictionarySource("resources/dictionaries/ad/jadni.tsv", "FLDNAME", "TEXT"))
    jadni.compute_embeddings(gpt4)

    pharmacog = MappingTable(MappingSource(AD_CDM_SRC, "PharmaCog", "CURIE"))
    pharmacog.add_descriptions(
        DataDictionarySource("resources/dictionaries/ad/pharmacog.csv", "Variable_Name", "Element_description"))
    pharmacog.compute_embeddings(gpt4)

    prevent_ad = MappingTable(MappingSource(AD_CDM_SRC, "PREVENT-AD", "CURIE"))
    prevent_ad.add_descriptions(
        DataDictionarySource("resources/dictionaries/ad/prevent-ad.csv", "variable", "description"))
    prevent_ad.compute_embeddings(gpt4)

    vita = MappingTable(MappingSource(AD_CDM_SRC, "VITA", "CURIE"))
    vita.add_descriptions(
        DataDictionarySource("resources/dictionaries/ad/vita.csv", "Variable_Name", "Element_description"))
    vita.compute_embeddings(gpt4)

    wmh_ad = MappingTable(MappingSource(AD_CDM_SRC, "VITA", "CURIE"))

    ad_datasets = [a4, abvib, adni, aibl, arwibo, dod_adni, edsd, emif, i_adni, jadni, pharmacog, prevent_ad, vita,
                   cdm_ad]
    ad_datasets_labels = ["A4", "Abvib", "ADNI", "AIBL", "ARWIBO", "DOD-ADNI", "EDSD", "EMIF", "I-ADNI", "JADNI",
                          "PharmaCog", "PREVENT-AD", "VITA", "AD-Mapper"]
    gpt_table, fuzzy_table = evaluate(ad_datasets, ad_datasets_labels)

    print("AD RESULTS:")
    print(gpt_table.to_string())
    print("-----------")
    print(fuzzy_table.to_string())
    print("-----------")

# embedding distribution
scatter_plot_two_distributions(pd_datesets, ad_datasets, "PD", "AD")
scatter_plot_all_cohorts(pd_datesets, ad_datasets, pd_datasets_labels, ad_datasets_labels)


