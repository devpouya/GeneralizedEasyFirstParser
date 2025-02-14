import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UD_PATH_RAW = 'ud/ud-treebanks-v2.5/'
UD_PATH_PROCESSED = 'ud/processed/'

UD_LANG_FOLDERS = {
    'af': 'UD_Afrikaans-AfriBooms/af_afribooms-ud-%s.conllu',
    # 'fr': 'UD_French-Sequoia/fr_sequoia-ud-%s.conllu',
    'no': 'UD_Norwegian-Bokmaal/no_bokmaal-ud-%s.conllu',
    'grc': 'UD_Ancient_Greek-Perseus/grc_perseus-ud-%s.conllu',
    # 'fr': 'UD_French-Spoken/fr_spoken-ud-%s.conllu',
    # 'no': 'UD_Norwegian-NynorskLIA/no_nynorsklia-ud-%s.conllu',
    # 'grc': 'UD_Ancient_Greek-PROIEL/grc_proiel-ud-%s.conllu',
    'gl': 'UD_Galician-CTG/gl_ctg-ud-%s.conllu',
    # 'no': 'UD_Norwegian-Nynorsk/no_nynorsk-ud-%s.conllu',
    # 'ar': 'UD_Arabic-NYUAD/ar_nyuad-ud-%s.conllu',
    # 'gl': 'UD_Galician-TreeGal/gl_treegal-ud-%s.conllu',
    'cu': 'UD_Old_Church_Slavonic-PROIEL/cu_proiel-ud-%s.conllu',
    'ar': 'UD_Arabic-PADT/ar_padt-ud-%s.conllu',
    #'de': 'UD_German-GSD/de_gsd-ud-%s.conllu',
    'de': 'UD_German-HDT/de_hdt-ud-%s.conllu',
    'fro': 'UD_Old_French-SRCMF/fro_srcmf-ud-%s.conllu',
    'hy': 'UD_Armenian-ArmTDP/hy_armtdp-ud-%s.conllu',
    # 'de': 'UD_German-HDT/de_hdt-ud-%s.conllu',
    'orv': 'UD_Old_Russian-TOROT/orv_torot-ud-%s.conllu',
    'eu': 'UD_Basque-BDT/eu_bdt-ud-%s.conllu',
    'got': 'UD_Gothic-PROIEL/got_proiel-ud-%s.conllu',
    'fa': 'UD_Persian-Seraji/fa_seraji-ud-%s.conllu',
    'be': 'UD_Belarusian-HSE/be_hse-ud-%s.conllu',
    'el': 'UD_Greek-GDT/el_gdt-ud-%s.conllu',
    # 'pl': 'UD_Polish-LFG/pl_lfg-ud-%s.conllu',
    'bg': 'UD_Bulgarian-BTB/bg_btb-ud-%s.conllu',
    'he': 'UD_Hebrew-HTB/he_htb-ud-%s.conllu',
    'pl': 'UD_Polish-PDB/pl_pdb-ud-%s.conllu',
    'bxr': 'UD_Buryat-BDT/bxr_bdt-ud-%s.conllu',
    'qhe': 'UD_Hindi_English-HIENCS/qhe_hiencs-ud-%s.conllu',
    # 'pt': 'UD_Portuguese-Bosque/pt_bosque-ud-%s.conllu',
    'ca': 'UD_Catalan-AnCora/ca_ancora-ud-%s.conllu',
    'hi': 'UD_Hindi-HDTB/hi_hdtb-ud-%s.conllu',
    'pt': 'UD_Portuguese-GSD/pt_gsd-ud-%s.conllu',
    # 'zh': 'UD_Chinese-GSDSimp/zh_gsdsimp-ud-%s.conllu',
    'hu': 'UD_Hungarian-Szeged/hu_szeged-ud-%s.conllu',
    # 'ro': 'UD_Romanian-Nonstandard/ro_nonstandard-ud-%s.conllu',
    'zh': 'UD_Chinese-GSD/zh_gsd-ud-%s.conllu',
    'id': 'UD_Indonesian-GSD/id_gsd-ud-%s.conllu',
    'ro': 'UD_Romanian-RRT/ro_rrt-ud-%s.conllu',
    'lzh': 'UD_Classical_Chinese-Kyoto/lzh_kyoto-ud-%s.conllu',
    'ga': 'UD_Irish-IDT/ga_idt-ud-%s.conllu',
    'ru': 'UD_Russian-GSD/ru_gsd-ud-%s.conllu',
    'cop': 'UD_Coptic-Scriptorium/cop_scriptorium-ud-%s.conllu',
    # 'it': 'UD_Italian-ISDT/it_isdt-ud-%s.conllu',
    # 'ru': 'UD_Russian-SynTagRus/ru_syntagrus-ud-%s.conllu',
    'hr': 'UD_Croatian-SET/hr_set-ud-%s.conllu',
    # 'it': 'UD_Italian-ParTUT/it_partut-ud-%s.conllu',
    # 'ru': 'UD_Russian-Taiga/ru_taiga-ud-%s.conllu',
    # 'it': 'UD_Italian-PoSTWITA/it_postwita-ud-%s.conllu',
    'gd': 'UD_Scottish_Gaelic-ARCOSG/gd_arcosg-ud-%s.conllu',
    # 'it': 'UD_Italian-TWITTIRO/it_twittiro-ud-%s.conllu',
    'sr': 'UD_Serbian-SET/sr_set-ud-%s.conllu',
    'it': 'UD_Italian-VIT/it_vit-ud-%s.conllu',
    'sk': 'UD_Slovak-SNK/sk_snk-ud-%s.conllu',
    'cs': 'UD_Czech-PDT/cs_pdt-ud-%s.conllu',
    # 'ja': 'UD_Japanese-BCCWJ/ja_bccwj-ud-%s.conllu',
    'sl': 'UD_Slovenian-SSJ/sl_ssj-ud-%s.conllu',
    'da': 'UD_Danish-DDT/da_ddt-ud-%s.conllu',
    'ja': 'UD_Japanese-GSD/ja_gsd-ud-%s.conllu',
    #'sl': 'UD_Slovenian-SST/sl_sst-ud-%s.conllu',
    'nl': 'UD_Dutch-Alpino/nl_alpino-ud-%s.conllu',
    'kk': 'UD_Kazakh-KTB/kk_ktb-ud-%s.conllu',
    # 'es': 'UD_Spanish-AnCora/es_ancora-ud-%s.conllu',
    # 'nl': 'UD_Dutch-LassySmall/nl_lassysmall-ud-%s.conllu',
    'ko': 'UD_Korean-GSD/ko_gsd-ud-%s.conllu',
    'es': 'UD_Spanish-GSD/es_gsd-ud-%s.conllu',
    # 'ko': 'UD_Korean-Kaist/ko_kaist-ud-%s.conllu',
    'sv': 'UD_Swedish-LinES/sv_lines-ud-%s.conllu',
    'en': 'UD_English-EWT/en_ewt-ud-%s.conllu',
    #'en': 'UD_English-GUM/en_gum-ud-%s.conllu',
    #'en': 'UD_English-EWT/en_ewt-ud-proj-%s.conllu',
    'kmr': 'UD_Kurmanji-MG/kmr_mg-ud-%s.conllu',
    'swl': 'UD_Swedish_Sign_Language-SSLC/swl_sslc-ud-%s.conllu',
    'la': 'UD_Latin-ITTB/la_ittb-ud-%s.conllu',
    # 'sv': 'UD_Swedish-Talbanken/sv_talbanken-ud-%s.conllu',
    # 'la': 'UD_Latin-Perseus/la_perseus-ud-%s.conllu',
    'ta': 'UD_Tamil-TTB/ta_ttb-ud-%s.conllu',
    # 'la': 'UD_Latin-PROIEL/la_proiel-ud-%s.conllu',
    'te': 'UD_Telugu-MTG/te_mtg-ud-%s.conllu',
    # 'et': 'UD_Estonian-EDT/et_edt-ud-%s.conllu',
    'lv': 'UD_Latvian-LVTB/lv_lvtb-ud-%s.conllu',
    'tr': 'UD_Turkish-IMST/tr_imst-ud-%s.conllu',
    'et': 'UD_Estonian-EWT/et_ewt-ud-%s.conllu',
    # 'lt': 'UD_Lithuanian-ALKSNIS/lt_alksnis-ud-%s.conllu',
    'uk': 'UD_Ukrainian-IU/uk_iu-ud-%s.conllu',
    # 'fi': 'UD_Finnish-FTB/fi_ftb-ud-%s.conllu',
    'lt': 'UD_Lithuanian-HSE/lt_hse-ud-%s.conllu',
    'hsb': 'UD_Upper_Sorbian-UFAL/hsb_ufal-ud-%s.conllu',
    'fi': 'UD_Finnish-TDT/fi_tdt-ud-%s.conllu',
    'olo': 'UD_Livvi-KKPP/olo_kkpp-ud-%s.conllu',
    'ur': 'UD_Urdu-UDTB/ur_udtb-ud-%s.conllu',
    'fr': 'UD_French-FTB/fr_ftb-ud-%s.conllu',
    'mt': 'UD_Maltese-MUDT/mt_mudt-ud-%s.conllu',
    'ug': 'UD_Uyghur-UDT/ug_udt-ud-%s.conllu',
    # 'fr': 'UD_French-GSD/fr_gsd-ud-%s.conllu',
    'mr': 'UD_Marathi-UFAL/mr_ufal-ud-%s.conllu',
    'vi': 'UD_Vietnamese-VTB/vi_vtb-ud-%s.conllu',
    # 'fr': 'UD_French-ParTUT/fr_partut-ud-%s.conllu',
    'sme': 'UD_North_Sami-Giella/sme_giella-ud-%s.conllu',
    'wo': 'UD_Wolof-WTB/wo_wtb-ud-%s.conllu',
    'yo': 'UD_Yoruba-YTB/yo_ytb-ud-%s.conllu',
}

UD_LANG_FNAMES = {name: '%s/%s' % (UD_PATH_RAW, folder) for name, folder in UD_LANG_FOLDERS.items()}



# actions


shift = "SHIFT"

reduce_l = "REDUCE_L"

reduce_r = "REDUCE_R"

reduce = "REDUCE"

left_arc_eager = "LEFT_ARC_EAGER"
left_arc_prime = "LEFT_ARC_PRIME"


right_arc_eager = "RIGHT_ARC_EAGER"
right_arc_prime = "RIGHT_ARC_PRIME"

left_arc_hybrid = "LEFT_ARC_H"

left_arc_2 = "LEFT_ARC_2"
right_arc_2 = "RIGHT_ARC_2"

right_attach = "ATTACH_RIGHT"
left_attach = "ATTACH_LEFT"

arc_standard = ([shift, reduce_l, reduce_r], list(range(3)))  # {shift: 0, reduce_l: 1, reduce_r: 2}
# arc_standard_actions = {0: fshift, 1: freduce_l, 2: freduce_r}

arc_eager = ([shift, left_arc_eager, right_arc_eager, reduce],
             list(range(4)))  # {shift: 0, left_arc_eager: 1, right_arc_eager: 2, reduce: 3}
# arc_eager_actions = {0: fshift, 1: fleft_arc_eager, 2: fright_arc_eager, 3: freduce}

hybrid = ([shift, left_arc_eager, reduce_r], list(range(3)))  # {shift: 0, left_arc_hybrid: 1, reduce_r: 2}

mh4 = ([shift, left_arc_eager, reduce_r, left_arc_prime, right_arc_prime,left_arc_2,right_arc_2], list(range(7)))
# hybrid_actions = {0: fshift, 1: fleft_arc, 2: freduce_r}

easy_first = ([left_attach,right_attach],list(range(2)))

agenda_std = (["LEFT","RIGHT"],list(range(2)))
agenda = "AGENDA-PARSER"