import os

from datastew.process.parsing import MappingSource, DataDictionarySource

from datastew.embedding import GPT4Adapter, MPNetAdapter
from datastew.evaluation import MatchingMethod, enrichment_analysis, evaluate
from datastew._mapping import _MappingTable
from dotenv import load_dotenv

from datastew.visualisation import enrichment_plot, scatter_plot_all_cohorts, scatter_plot_two_distributions

EVAL_PD = True
EVAL_AD = True

# Constants for file paths
PD_CDM_SRC = "resources/cdm_pd.xlsx"
AD_CDM_SRC = "resources/cdm_ad.csv"
CDM_CURIE_CSV = "resources/cdm_curie.csv"
LRRK2_DICT_XLSX = "resources/dictionaries/pd/LRRK2.xlsx"
OPDC_DICT_CSV = "resources/dictionaries/pd/OPDC.csv"
TPD_DICT_CSV = "resources/dictionaries/pd/TPD.csv"
A4_DICT_CSV = "resources/dictionaries/ad/a4.csv"
ABVIB_DICT_CSV = "resources/dictionaries/ad/abvib.csv"
ADNI_DICT_CSV = "resources/dictionaries/ad/ADNIMERGE_DICT_27Nov2023 2.csv"
AIBL_DICT_CSV = "resources/dictionaries/ad/aibl.csv"
ARWIBO_DICT_CSV = "resources/dictionaries/ad/arwibo.csv"
DOD_ADNI_DICT_CSV = "resources/dictionaries/ad/dod-adni.csv"
EDSD_DICT_XLSX = "resources/dictionaries/ad/edsd.xlsx"
EMIF_DICT_XLSX = "resources/dictionaries/ad/emif.xlsx"
I_ADNI_DICT_CSV = "resources/dictionaries/ad/i-adni.csv"
JADNI_DICT_TSV = "resources/dictionaries/ad/jadni.tsv"
PHARMACOG_DICT_CSV = "resources/dictionaries/ad/pharmacog.csv"
PREVENT_AD_DICT_CSV = "resources/dictionaries/ad/prevent-ad.csv"
VITA_DICT_CSV = "resources/dictionaries/ad/vita.csv"
PPMI_DICT_SRC = "resources/dictionaries/pd/ppmi.csv"
LUXPARK_DICT_SRC = "resources/dictionaries/pd/luxpark.xlsx"
BIOFIND_DICT_SRC = "resources/dictionaries/pd/biofind.csv"

load_dotenv()
gpt4 = GPT4Adapter(api_key=os.getenv("GPT_KEY"))
mpnet = MPNetAdapter()

# PD Mappings

if EVAL_PD:
    cdm_pd_gpt = _MappingTable(MappingSource(PD_CDM_SRC, "Feature", "CURIE"))
    cdm_pd_gpt.joined_mapping_table["identifier"].to_csv("resources/cdm_curie.csv", index=False)
    cdm_pd_gpt.add_descriptions(DataDictionarySource(PD_CDM_SRC, "Feature", "Definition"))
    cdm_pd_gpt.compute_embeddings(gpt4)

    cdm_pd_mpnet = _MappingTable(MappingSource(PD_CDM_SRC, "Feature", "CURIE"))
    cdm_pd_mpnet.joined_mapping_table["identifier"].to_csv("resources/cdm_curie.csv", index=False)
    cdm_pd_mpnet.add_descriptions(DataDictionarySource(PD_CDM_SRC, "Feature", "Definition"))
    cdm_pd_mpnet.compute_embeddings(mpnet)

    ppmi_gpt = _MappingTable(MappingSource(PD_CDM_SRC, "PPMI", "CURIE"))
    ppmi_gpt.add_descriptions(DataDictionarySource(PPMI_DICT_SRC, "ITM_NAME", "DSCR"))
    ppmi_gpt.compute_embeddings(gpt4)

    ppmi_mpnet = _MappingTable(MappingSource(PD_CDM_SRC, "PPMI", "CURIE"))
    ppmi_mpnet.add_descriptions(DataDictionarySource(PPMI_DICT_SRC, "ITM_NAME", "DSCR"))
    ppmi_mpnet.compute_embeddings(mpnet)

    luxpark_gpt = _MappingTable(MappingSource(PD_CDM_SRC, "LuxPARK", "CURIE"))
    luxpark_gpt.add_descriptions(DataDictionarySource(LUXPARK_DICT_SRC, "Variable / Field Name", "Field Label"))
    luxpark_gpt.compute_embeddings(gpt4)

    luxpark_mpnet = _MappingTable(MappingSource(PD_CDM_SRC, "LuxPARK", "CURIE"))
    luxpark_mpnet.add_descriptions(DataDictionarySource(LUXPARK_DICT_SRC, "Variable / Field Name", "Field Label"))
    luxpark_mpnet.compute_embeddings(mpnet)

    biofind_gpt = _MappingTable(MappingSource(PD_CDM_SRC, "BIOFIND", "CURIE"))
    biofind_gpt.add_descriptions(DataDictionarySource(BIOFIND_DICT_SRC, "ITM_NAME", "DSCR"))
    biofind_gpt.compute_embeddings(gpt4)

    biofind_mpnet = _MappingTable(MappingSource(PD_CDM_SRC, "BIOFIND", "CURIE"))
    biofind_mpnet.add_descriptions(DataDictionarySource(BIOFIND_DICT_SRC, "ITM_NAME", "DSCR"))
    biofind_mpnet.compute_embeddings(mpnet)

    lrrk2_gpt = _MappingTable(MappingSource(PD_CDM_SRC, "LRRK2", "CURIE"))
    lrrk2_gpt.add_descriptions(DataDictionarySource("resources/dictionaries/pd/LRRK2.xlsx", "Variable", "Label"))
    lrrk2_gpt.compute_embeddings(gpt4)

    lrrk2_mpnet = _MappingTable(MappingSource(PD_CDM_SRC, "LRRK2", "CURIE"))
    lrrk2_mpnet.add_descriptions(DataDictionarySource("resources/dictionaries/pd/LRRK2.xlsx", "Variable", "Label"))
    lrrk2_mpnet.compute_embeddings(mpnet)

    opdc_gpt = _MappingTable(MappingSource(PD_CDM_SRC, "OPDC", "CURIE"))
    opdc_gpt.add_descriptions(DataDictionarySource("resources/dictionaries/pd/OPDC.csv", "Variable Name", "Variable description"))
    opdc_gpt.compute_embeddings(gpt4)

    opdc_mpnet = _MappingTable(MappingSource(PD_CDM_SRC, "OPDC", "CURIE"))
    opdc_mpnet.add_descriptions(DataDictionarySource("resources/dictionaries/pd/OPDC.csv", "Variable Name", "Variable description"))
    opdc_mpnet.compute_embeddings(mpnet)

    tpd_gpt = _MappingTable(MappingSource(PD_CDM_SRC, "TPD", "CURIE"))
    tpd_gpt.add_descriptions(DataDictionarySource("resources/dictionaries/pd/TPD.csv", "Variable Name", "Variable description"))
    tpd_gpt.compute_embeddings(gpt4)

    tpd_mpnet = _MappingTable(MappingSource(PD_CDM_SRC, "TPD", "CURIE"))
    tpd_mpnet.add_descriptions(DataDictionarySource("resources/dictionaries/pd/TPD.csv", "Variable Name", "Variable description"))
    tpd_mpnet.compute_embeddings(mpnet)

    pd_datasets_gpt = [opdc_gpt, tpd_gpt, biofind_gpt, lrrk2_gpt, luxpark_gpt, ppmi_gpt, cdm_pd_gpt]
    pd_datasets_mpnet = [opdc_mpnet, tpd_mpnet, biofind_mpnet, lrrk2_mpnet, luxpark_mpnet, ppmi_mpnet, cdm_pd_mpnet]
    pd_datasets_labels = ["OPDC", "PRoBaND", "BIOFIND", "LCC", "LuxPARK", "PPMI", "PASSIONATE"]

    # enrichment analysis
    luxpark_passionate_enrichment_gpt = enrichment_analysis(luxpark_gpt, cdm_pd_gpt, 20, MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE)
    luxpark_passionate_enrichment_mpnet = enrichment_analysis(luxpark_mpnet, cdm_pd_mpnet, 20, MatchingMethod.COSINE_EMBEDDING_DISTANCE)
    luxpark_passionate_enrichment_fuzzy = enrichment_analysis(luxpark_gpt, cdm_pd_gpt, 20, MatchingMethod.FUZZY_STRING_MATCHING)
    label1 = "Enrichment Plot LuxPARK to CDM"
    
    ppmi_passionate_enrichment_gpt = enrichment_analysis(ppmi_gpt, cdm_pd_gpt, 20, MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE)
    ppmi_passionate_enrichment_mpnet = enrichment_analysis(ppmi_mpnet, cdm_pd_mpnet, 20, MatchingMethod.COSINE_EMBEDDING_DISTANCE)
    ppmi_passionate_enrichment_fuzzy = enrichment_analysis(ppmi_gpt, cdm_pd_gpt, 20, MatchingMethod.FUZZY_STRING_MATCHING)
    label2 = "Enrichment Plot PPMI to CDM"

    ppmi_luxpark_enrichment_gpt = enrichment_analysis(ppmi_gpt, luxpark_gpt, 20, MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE)
    ppmi_luxpark_enrichment_mpnet = enrichment_analysis(ppmi_mpnet, luxpark_mpnet, 20, MatchingMethod.COSINE_EMBEDDING_DISTANCE)
    ppmi_luxpark_enrichment_fuzzy = enrichment_analysis(ppmi_gpt, luxpark_gpt, 20, MatchingMethod.FUZZY_STRING_MATCHING)
    label3 = "Enrichment Plot PPMI to LuxPARK"

    enrichment_plot(luxpark_passionate_enrichment_gpt, luxpark_passionate_enrichment_mpnet, luxpark_passionate_enrichment_fuzzy, label1, save_plot=True)
    enrichment_plot(ppmi_passionate_enrichment_gpt, ppmi_passionate_enrichment_mpnet, ppmi_passionate_enrichment_fuzzy, label2, save_plot=True)
    enrichment_plot( ppmi_luxpark_enrichment_gpt, ppmi_luxpark_enrichment_mpnet, ppmi_luxpark_enrichment_fuzzy, label3, save_plot=True)
    
    print(luxpark_passionate_enrichment_gpt)
    print(luxpark_passionate_enrichment_mpnet)
    print(luxpark_passionate_enrichment_fuzzy)
    print(ppmi_passionate_enrichment_gpt)
    print(ppmi_passionate_enrichment_mpnet)
    print(ppmi_passionate_enrichment_fuzzy)
    print(ppmi_luxpark_enrichment_gpt)
    print(ppmi_luxpark_enrichment_mpnet)
    print(ppmi_luxpark_enrichment_fuzzy)

    gpt_table1 = evaluate(pd_datasets_gpt, pd_datasets_labels, store_results=True)
    fuzzy_table1 = evaluate(pd_datasets_gpt, pd_datasets_labels, store_results=True, model="fuzzy")
    mpnet_table1 = evaluate(pd_datasets_mpnet, pd_datasets_labels, store_results=True, model="mpnet")

    print("PD RESULTS:")
    print("GPT")
    print("-----------")
    print(gpt_table1)
    print("-----------")
    print("MPNet")
    print("-----------")
    print(mpnet_table1)
    print("-----------")
    print("Fuzzy")
    print("-----------")
    print(fuzzy_table1)
    print("-----------")

# AD Mappings

if EVAL_AD:
    cdm_ad_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "Feature", "CDM"))
    cdm_ad_gpt.add_descriptions(DataDictionarySource(AD_CDM_SRC, "Feature", "Definition"))
    cdm_ad_gpt.compute_embeddings(gpt4)

    cdm_ad_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "Feature", "CDM"))
    cdm_ad_mpnet.add_descriptions(DataDictionarySource(AD_CDM_SRC, "Feature", "Definition"))
    cdm_ad_mpnet.compute_embeddings(mpnet)

    abvib_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "abvib", "CDM"))
    abvib_gpt.add_descriptions(DataDictionarySource(ABVIB_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    abvib_gpt.compute_embeddings(gpt4)

    abvib_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "abvib", "CDM"))
    abvib_mpnet.add_descriptions(DataDictionarySource(ABVIB_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    abvib_mpnet.compute_embeddings(mpnet)

    adni_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "adni", "CDM"))
    adni_gpt.add_descriptions(DataDictionarySource(ADNI_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    adni_gpt.compute_embeddings(gpt4)

    adni_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "adni", "CDM"))
    adni_mpnet.add_descriptions(DataDictionarySource(ADNI_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    adni_mpnet.compute_embeddings(mpnet)

    a4_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "a4", "CDM"))
    a4_gpt.add_descriptions(DataDictionarySource(A4_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    a4_gpt.compute_embeddings(gpt4)

    a4_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "a4", "CDM"))
    a4_mpnet.add_descriptions(DataDictionarySource(A4_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    a4_mpnet.compute_embeddings(mpnet)

    aibl_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "aibl", "CDM"))
    aibl_gpt.add_descriptions(DataDictionarySource(AIBL_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    aibl_gpt.compute_embeddings(gpt4)

    aibl_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "aibl", "CDM"))
    aibl_mpnet.add_descriptions(DataDictionarySource(AIBL_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    aibl_mpnet.compute_embeddings(mpnet)

    arwibo_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "arwibo", "CDM"))
    arwibo_gpt.add_descriptions(DataDictionarySource(ARWIBO_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    arwibo_gpt.compute_embeddings(gpt4)

    arwibo_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "arwibo", "CDM"))
    arwibo_mpnet.add_descriptions(DataDictionarySource(ARWIBO_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    arwibo_mpnet.compute_embeddings(mpnet)

    dod_adni_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "dod-adni", "CDM"))
    dod_adni_gpt.add_descriptions(DataDictionarySource(DOD_ADNI_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    dod_adni_gpt.compute_embeddings(gpt4)

    dod_adni_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "dod-adni", "CDM"))
    dod_adni_mpnet.add_descriptions(DataDictionarySource(DOD_ADNI_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    dod_adni_mpnet.compute_embeddings(mpnet)

    edsd_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "edsd", "CDM"))
    edsd_gpt.add_descriptions(DataDictionarySource(EDSD_DICT_XLSX, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    edsd_gpt.compute_embeddings(gpt4)

    edsd_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "edsd", "CDM"))
    edsd_mpnet.add_descriptions(DataDictionarySource(EDSD_DICT_XLSX, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    edsd_mpnet.compute_embeddings(mpnet)

    emif_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "emif", "CDM"))
    emif_gpt.add_descriptions(DataDictionarySource(EMIF_DICT_XLSX, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    emif_gpt.compute_embeddings(gpt4)

    emif_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "emif", "CDM"))
    emif_mpnet.add_descriptions(DataDictionarySource(EMIF_DICT_XLSX, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    emif_mpnet.compute_embeddings(mpnet)

    iadni_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "i-adni", "CDM"))
    iadni_gpt.add_descriptions(DataDictionarySource(I_ADNI_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    iadni_gpt.compute_embeddings(gpt4)

    iadni_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "i-adni", "CDM"))
    iadni_mpnet.add_descriptions(DataDictionarySource(I_ADNI_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    iadni_mpnet.compute_embeddings(mpnet)

    jadni_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "jadni", "CDM"))
    jadni_gpt.add_descriptions(DataDictionarySource(JADNI_DICT_TSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    jadni_gpt.compute_embeddings(gpt4)

    jadni_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "jadni", "CDM"))
    jadni_mpnet.add_descriptions(DataDictionarySource(JADNI_DICT_TSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    jadni_mpnet.compute_embeddings(mpnet)

    pharmacog_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "pharmacog", "CDM"))
    pharmacog_gpt.add_descriptions(DataDictionarySource(PHARMACOG_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    pharmacog_gpt.compute_embeddings(gpt4)

    pharmacog_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "pharmacog", "CDM"))
    pharmacog_mpnet.add_descriptions(DataDictionarySource(PHARMACOG_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    pharmacog_mpnet.compute_embeddings(mpnet)

    prevent_ad_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "prevent-ad", "CDM"))
    prevent_ad_gpt.add_descriptions(DataDictionarySource(PREVENT_AD_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    prevent_ad_gpt.compute_embeddings(gpt4)

    prevent_ad_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "prevent-ad", "CDM"))
    prevent_ad_mpnet.add_descriptions(DataDictionarySource(PREVENT_AD_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    prevent_ad_mpnet.compute_embeddings(mpnet)

    vita_gpt = _MappingTable(MappingSource(AD_CDM_SRC, "vita", "CDM"))
    vita_gpt.add_descriptions(DataDictionarySource(VITA_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    vita_gpt.compute_embeddings(gpt4)

    vita_mpnet = _MappingTable(MappingSource(AD_CDM_SRC, "vita", "CDM"))
    vita_mpnet.add_descriptions(DataDictionarySource(VITA_DICT_CSV, "FIELD_LABEL", "FIELD_DESCRIPTION"))
    vita_mpnet.compute_embeddings(mpnet)

    ad_datasets_gpt = [a4_gpt, abvib_gpt, adni_gpt, aibl_gpt, arwibo_gpt, dod_adni_gpt, edsd_gpt, emif_gpt, iadni_gpt, jadni_gpt,
                       pharmacog_gpt, prevent_ad_gpt, vita_gpt, cdm_ad_gpt]
    ad_datasets_mpnet = [a4_mpnet, abvib_mpnet, adni_mpnet, aibl_mpnet, arwibo_mpnet, dod_adni_mpnet, edsd_mpnet, emif_mpnet,
                         iadni_mpnet, jadni_mpnet, pharmacog_mpnet, prevent_ad_mpnet, vita_mpnet, cdm_ad_mpnet]
    ad_datasets_labels = ["A4", "ABVIB", "ADNI", "AIBL", "ARWIBO", "DOD-ADNI", "EDSD", "EMIF", "I-ADNI", "JADNI", "PharmaCog",
                          "PREVENT-AD", "VITA", "AD-Mapper"]
    gpt_table2 = evaluate(ad_datasets_gpt, ad_datasets_labels, store_results=True, results_root_dir="resources/results/ad")
    fuzzy_table2 = evaluate(ad_datasets_gpt, ad_datasets_labels, store_results=True, model="fuzzy", results_root_dir="resources/results/ad")
    mpnet_table2 = evaluate(ad_datasets_mpnet, ad_datasets_labels, store_results=True, model="mpnet", results_root_dir="resources/results/ad")

    print("AD RESULTS:")
    print("GPT")
    print("-----------")
    print(gpt_table2.to_string())
    print("-----------")
    print("MPNet")
    print("-----------")
    print(mpnet_table2.to_string())
    print("-----------")
    print("Fuzzy")
    print("-----------")
    print(fuzzy_table2.to_string())
    print("-----------")

# embedding distribution
scatter_plot_two_distributions(pd_datasets_gpt, ad_datasets_gpt, "PD", "AD")
scatter_plot_all_cohorts(pd_datasets_gpt, ad_datasets_gpt, pd_datasets_labels, ad_datasets_labels)
