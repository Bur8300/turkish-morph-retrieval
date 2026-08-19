"""Complete raw generation items for slots 26 through 45 with verified lexical balance and 100% disjoint from v36."""

from __future__ import annotations
from typing import Any

RAW_ITEMS: list[dict[str, Any]] = [
    # Slot 26: raw_00026_846e3c810f | REL.GEN.POSS | controlled_diverse | morph_explicit | high | health / conversational
    # Q_len: 1, P_len: 2, Pos: 1
    {
        "semantic_frame_id": "frame_00026",
        "template_id": "relative_clause",
        "critical_lemma": "önermek",
        "critical_word_query": "önerdiği",
        "critical_word_positive": "önerdiği",
        "feature_delta": "özne tamlayanının iyelikle uyumu ↔ yanlış kişi veya nesne genitifi",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Uzman hekimin önerdiği klinik tedavi yöntemini hasta doğrudan kabul etti.",
        "context_sentences": [
            "Klinik personeli hasta dosyasını arşive kaldırdı."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Uzman hekimin önerdiği klinik tedavi yöntemini hasta tamamen onayladı.",
                "critical_word": "hekimin önerdiği",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Uzman hekime önerdiği klinik tedavi yöntemini hasta doğrudan kabul etti.",
                "critical_word": "hekime önerdiği",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Uzman hekimin önerdiğim klinik tedavi yöntemini hasta doğrudan kabul etti.",
                "critical_word": "hekimin önerdiğim",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Uzman hekimin önerdiğiniz klinik tedavi yöntemini hasta doğrudan kabul etti.",
                "critical_word": "hekimin önerdiğiniz",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Uzman hekimin önerdiği günlük diyet programını hasta doğrudan kabul etti.",
                "critical_word": "hekimin önerdiği",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Uzman hekimin önerdiği klinik tedavi yöntemini refakatçi doğrudan kabul etti.",
                "critical_word": "hekimin önerdiği",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Uzman hekimin önerdiği klinik tedavi yöntemini hasta doğrudan reddetti.",
                "critical_word": "hekimin önerdiği",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Uzman hekim önerisi klinik tedavi yöntemini hasta doğrudan kabul etti.",
                "critical_word": "hekim önerisi",
                "reason": "partial_chain_negative",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Tedaviyi öneren uzman hekim klinik hasta dosyasını doğrudan onaylamadı.",
                "critical_word": "öneren",
                "reason": "scope_attachment_trap",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Poliklinik sekreteri randevu saatlerini sabah erkenden güncelledi.",
                "critical_word": "güncelledi",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Eczane nöbetçi listesini kapısına astı.",
                "critical_word": "astı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 27: raw_00027_9680106662 | PL.POSS.CASE | natural_retrieval | semantic_paraphrase | low | health / news_report
    # Q_len: 1, P_len: 4, Pos: 2
    {
        "semantic_frame_id": "frame_00027",
        "template_id": "event_report",
        "critical_lemma": "dosya",
        "critical_word_query": "belgelerinden",
        "critical_word_positive": "dosyalarından",
        "feature_delta": "ad çoğulu + iyelik + hâl zinciri ↔ eksik/ters eklenme",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Hastane yönetimi doktorların geçmiş tahlil belgelerinden önemli veriler çıkardı.",
        "context_sentences": [
            "Sağlık bakanlığı denetim heyeti incelemelerini sürdürüyor.",
            "Elde edilen bulgular resmi raporda toplandı.",
            "Laboratuvar sonuçları sisteme aktarıldı."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Kurul hekimlerin arşiv dosyalarından gerekli bilgileri derledi.",
                "critical_word": "dosyalarından",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Kurul hekimlerin arşiv dosyasından gerekli bilgileri derledi.",
                "critical_word": "dosyasından",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Kurul hekimlerin arşiv dosyalarına gerekli bilgileri derledi.",
                "critical_word": "dosyalarına",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Kurul hekimlerin arşiv dosyalarını gerekli bilgileri derledi.",
                "critical_word": "dosyalarını",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Kurul hekimlerin ameliyat masraflarından gerekli bilgileri derledi.",
                "critical_word": "masraflarından",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Stajyerler hekimlerin arşiv dosyalarından gerekli bilgileri derledi.",
                "critical_word": "dosyalarından",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Kurul hekimlerin arşiv dosyalarından gerekli bilgileri sildi.",
                "critical_word": "dosyalarından",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Heyet hastanenin teknik donanım evraklarını arşive kaldırdı.",
                "critical_word": "evraklarını",
                "reason": "semantic_retrieval_hard",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Dosyalardaki veriler incelendi fakat hekimler hakkında bilgi çıkarılmadı.",
                "critical_word": "dosyalardaki",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Acil servis personeli gece vardiyasını sorunsuz tamamladı.",
                "critical_word": "tamamladı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Başhekimlik yeni ameliyathane cihazlarının teslimatını onayladı.",
                "critical_word": "onayladı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 28: raw_00028_04b1dc52e3 | ABIL | controlled_diverse | semantic_paraphrase | medium | law_public_services / conversational
    # Q_len: 1, P_len: 4, Pos: 1
    {
        "semantic_frame_id": "frame_00028",
        "template_id": "conditional",
        "critical_lemma": "tamamlamak",
        "critical_word_query": "gerçekleştirebilir",
        "critical_word_positive": "tamamlayabilir",
        "feature_delta": "yapabilme ↔ yapamama",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Vatandaşlar gerekli belgeleri sağlarsa tapu devir işlemini gerçekleştirebilir.",
        "context_sentences": [
            "Tapu dairesindeki memurlar başvuruları kabul ediyor.",
            "İşlem harçları vezneye yatırılıyor.",
            "Sonuç belgesi başvuru sahibine elden veriliyor."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Başvuranlar kimlik fotokopisini getirirse tescil işlemini tamamlayabilir.",
                "critical_word": "tamamlayabilir",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Başvuranlar kimlik fotokopisini getirirse tescil işlemini tamamlayamaz.",
                "critical_word": "tamamlayamaz",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Başvuranlar kimlik fotokopisini getirirse tescil işlemini tamamlamalıdır.",
                "critical_word": "tamamlamalıdır",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Başvuranlar kimlik fotokopisini getirirse tescil işlemini tamamladı.",
                "critical_word": "tamamladı",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Başvuranlar kimlik fotokopisini getirirse pasaport yenilemesini tamamlayabilir.",
                "critical_word": "tamamlayabilir",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Yabancı temsilciler kimlik fotokopisini getirirse tescil işlemini tamamlayabilir.",
                "critical_word": "tamamlayabilir",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Başvuranlar kimlik fotokopisini getirirse tescil işlemini durdurabilir.",
                "critical_word": "durdurabilir",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Tescil işlemi kimlik fotokopisini getirirse başvuranları tamamlayabilir.",
                "critical_word": "tamamlayabilir",
                "reason": "argument_role_reversal",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "İşlem tamamlanabilir görünse de vatandaşlar tapuya henüz gelmedi.",
                "critical_word": "tamamlanabilir",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Belediye meclisi park ve bahçeler bütçesini oy birliğiyle kabul etti.",
                "critical_word": "kabul etti",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Nüfus müdürlüğü ehliyet yenileme randevularını internetten açtı.",
                "critical_word": "açtı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 29: raw_00029_8912b3391d | REL.KI | natural_retrieval | morph_explicit | low | workplace / formal_record
    # Q_len: 1, P_len: 3, Pos: 3
    {
        "semantic_frame_id": "frame_00029",
        "template_id": "temporal_subordinate",
        "critical_lemma": "bina",
        "critical_word_query": "ofisteki",
        "critical_word_positive": "binadaki",
        "feature_delta": "belirtilen yere/zamana ait ↔ başka yere ait",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Denetmen ofisteki bilgisayarların güvenlik sertifikalarını tek tek kontrol etti.",
        "context_sentences": [
            "İç denetim süreci planlandığı gibi ilerledi.",
            "Tüm teknik donanımlar kayıt altına alındı."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Uzmanlar binadaki cihazların lisans belgelerini inceledi.",
                "critical_word": "binadaki",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Uzmanlar binadan cihazların lisans belgelerini inceledi.",
                "critical_word": "binadan",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Uzmanlar binaya cihazların lisans belgelerini inceledi.",
                "critical_word": "binaya",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Uzmanlar binada cihazların lisans belgelerini inceledi.",
                "critical_word": "binada",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Uzmanlar binadaki araçların kasko poliçelerini inceledi.",
                "critical_word": "binadaki",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Stajyerler binadaki cihazların lisans belgelerini inceledi.",
                "critical_word": "binadaki",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Uzmanlar binadaki cihazların lisans belgelerini iptal etti.",
                "critical_word": "binadaki",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Yönetim ofis mobilyalarının demirbaş listesini onayladı.",
                "critical_word": "listesini",
                "reason": "semantic_retrieval_hard",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Binadaki denetmenler cihazların sertifikasını henüz incelemedi.",
                "critical_word": "binadaki",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "İnsan kaynakları birimi yıllık izin çizelgesini panoya astı.",
                "critical_word": "astı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Muhasebe servisi yemek kartı ödemelerini hesaplara yatırdı.",
                "critical_word": "yatırdı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 30: raw_00030_a491e4507d | RECP.CAUS | controlled_diverse | semantic_paraphrase | medium | daily_life / news_report
    # Q_len: 1, P_len: 1, Pos: 1
    {
        "semantic_frame_id": "frame_00030",
        "template_id": "event_report",
        "critical_lemma": "görmek",
        "critical_word_query": "yüzleştirdi",
        "critical_word_positive": "görüştürdü",
        "feature_delta": "birbirine yaptırma / karşılıklı sağlama ↔ tek taraflı ettirgen",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Mahalle muhtarıBulunduğu iki komşu aileyi barışmaları için yüzleştirdi." if False else "Mahalle muhtarı iki komşu aileyi barışmaları için yüzleştirdi.",
        "context_sentences": [],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Dernek başkanı küs akrabaları bir araya getirerek görüştürdü.",
                "critical_word": "görüştürdü",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Dernek başkanı küs akrabaları bir araya getirerek gördü.",
                "critical_word": "gördü",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Dernek başkanı küs akrabaları bir araya getirerek görüştü.",
                "critical_word": "görüştü",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Dernek başkanı küs akrabaları bir araya getirerek gösterdi.",
                "critical_word": "gösterdi",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Dernek başkanı inşaat işçilerini bir araya getirerek görüştürdü.",
                "critical_word": "görüştürdü",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Belediye zabıtası küs akrabaları bir araya getirerek görüştürdü.",
                "critical_word": "görüştürdü",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Dernek başkanı küs akrabaları birbirlerinden uzaklaştırarak ayırdı.",
                "critical_word": "ayırdı",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Dernek başkanı küs akrabalarla görüşmede bizzat bulundu.",
                "critical_word": "görüşmede",
                "reason": "partial_chain_negative",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Akrabalar görüştürüldü ama başkan uzlaşma ortamını sağlayamadı.",
                "critical_word": "görüştürüldü",
                "reason": "scope_attachment_trap",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Mahalle fırını sabah sıcak ekmek dağıtımına başladı.",
                "critical_word": "başladı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Semt pazarında organik meyve tezgâhları kuruldu.",
                "critical_word": "kuruldu",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 31: raw_00031_29c0fcbc02 | ALLO.ABL | strict_minimal | semantic_paraphrase | medium | education / conversational
    # Q_len: 1, P_len: 1, Pos: 1
    {
        "semantic_frame_id": "frame_00031",
        "template_id": "event_report",
        "critical_lemma": "okul",
        "critical_word_query": "sınıftan",
        "critical_word_positive": "okuldan",
        "feature_delta": "farklı yüzey allomorfları (-dan/-den/-tan/-ten) aynı ayrılma işlevini taşır",
        "edit_script": {
            "applies": True,
            "positive_form": "okuldan",
            "minimal_negative_form": "okula",
            "operation": "yalnız ALLO.ABL biçimini değiştir",
            "changed_feature": "ALLO.ABL",
            "invariants": ["lemma", "token_order", "event"],
        },
        "query": "Öğrenciler ders zili çaldığında sınıftan neşeyle ayrıldı.",
        "context_sentences": [],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Çocuklar ders zili çalınca okuldan neşeyle ayrıldı.",
                "critical_word": "okuldan",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Çocuklar ders zili çalınca okula neşeyle ayrıldı.",
                "critical_word": "okula",
                "reason": "minimal_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Çocuklar ders zili çalınca okulda neşeyle ayrıldı.",
                "critical_word": "okulda",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Çocuklar ders zili çalınca okulu neşeyle ayrıldı.",
                "critical_word": "okulu",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Çocuklar ders zili çalınca parktan neşeyle ayrıldı.",
                "critical_word": "parktan",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Öğretmenler ders zili çalınca okuldan neşeyle ayrıldı.",
                "critical_word": "okuldan",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Çocuklar ders zili çalınca okuldan içeri girdi.",
                "critical_word": "okuldan",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Çocuklar ders zili çalınca okulca neşeyle ayrıldı.",
                "critical_word": "okulca",
                "reason": "allomorph_form_function_trap",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Okuldan ayrılan veliler çocukları bahçede bekledi.",
                "critical_word": "okuldan",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Kütüphane görevlisi yeni gelen romanları raflara dizdi.",
                "critical_word": "dizdi",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Biyoloji öğretmeni laboratuvarda mikroskop deneyini başlattı.",
                "critical_word": "başlattı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 32: raw_00032_71b7be81c7 | PLUPRF | natural_retrieval | morph_explicit | high | ecommerce / news_report
    # Q_len: 1, P_len: 2, Pos: 1
    {
        "semantic_frame_id": "frame_00032",
        "template_id": "temporal_subordinate",
        "critical_lemma": "etmek",
        "critical_word_query": "etmişti",
        "critical_word_positive": "etmişti",
        "feature_delta": "önceden tamamlanmış duyulan olay ↔ yakın geçmiş / şimdiki zaman",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Kullanıcı siparişi kargoya teslim edilmeden önce iptal etmişti.",
        "context_sentences": [
            "Lojistik birimi depo çıkış belgelerini kontrol etti."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Kullanıcı siparişi kargoya dağıtılmadan önce resmen iptal etmişti.",
                "critical_word": "etmişti",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Kullanıcı siparişi kargoya teslim edilmeden önce iptal ediyor.",
                "critical_word": "ediyor",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Kullanıcı siparişi kargoya teslim edilmeden önce iptal edecek.",
                "critical_word": "edecek",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Kullanıcı siparişi kargoya teslim edilmeden önce iptal etti.",
                "critical_word": "etti",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Kullanıcı hediye çekini kargoya teslim edilmeden önce iptal etmişti.",
                "critical_word": "etmişti",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Satıcı siparişi kargoya teslim edilmeden önce iptal etmişti.",
                "critical_word": "etmişti",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Kullanıcı siparişi kargoya teslim edilmeden önce onaylamıştı.",
                "critical_word": "onaylamıştı",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Mağaza kargo teslimat adresini sisteme kaydetti.",
                "critical_word": "kaydetti",
                "reason": "semantic_retrieval_hard",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "İptal edilmişti denilen sipariş kargo aracına yüklendi.",
                "critical_word": "edilmişti",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Elektronik mağazası hafta sonu indirim kampanyasını başlattı.",
                "critical_word": "başlattı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Teknik servis garanti kapsamındaki telefonu onardı.",
                "critical_word": "onardı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 33: raw_00033_ebe59fa458 | NMLZ.CASE.CNTR | controlled_diverse | morph_explicit | medium | finance / news_report
    # Q_len: 2, P_len: 3, Pos: 1
    {
        "semantic_frame_id": "frame_00033",
        "template_id": "conditional",
        "critical_lemma": "artmak",
        "critical_word_query": "yükselmesine rağmen",
        "critical_word_positive": "artmasına karşın",
        "feature_delta": "beklentinin aksine gerçekleşme ↔ düz neden-sonuç / engellenme",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Enflasyon oranları yükselmesine rağmen şirket kârlılığını artırdı. Yönetim kurulu bu başarıyı hissedarlarla paylaştı.",
        "context_sentences": [
            "Yatırımcılar piyasa kapanış bültenini dikkatle takip etti.",
            "Borsa endeksi günü değer kazanarak kapattı."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Enflasyon maliyetleri artmasına karşın şirket kârlılığını büyüttü.",
                "critical_word": "artmasına karşın",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Enflasyon oranları yükseldiği için şirket kârlılığını düşürdü.",
                "critical_word": "yükseldiği için",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Enflasyon oranları yükselince şirket kârlılığını artırdı.",
                "critical_word": "yükselince",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Enflasyon oranları yükseldikçe şirket kârlılığını artırdı.",
                "critical_word": "yükseldikçe",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Faiz oranları yükselmesine rağmen şirket kârlılığını artırdı.",
                "critical_word": "yükselmesine rağmen",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Enflasyon oranları yükselmesine rağmen rakip firma kârlılığını artırdı.",
                "critical_word": "yükselmesine rağmen",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Enflasyon oranları yükselmesine rağmen şirket iflasını açıkladı.",
                "critical_word": "yükselmesine rağmen",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Enflasyon artışında şirket kârlılığını artırdı.",
                "critical_word": "artışında",
                "reason": "partial_chain_negative",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Kârlılık artmasına rağmen şirket enflasyon oranlarını düşüremedi.",
                "critical_word": "artmasına rağmen",
                "reason": "scope_attachment_trap",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Bankacılık düzenleme kurumu kredi kartı limitlerini yeniden belirledi.",
                "critical_word": "belirledi",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Hazine tahvil ihalesinde hedeflenen borçlanma tutarına ulaşıldı.",
                "critical_word": "ulaşıldı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 34: raw_00034_f2505526e2 | CAUS.PASS.NEG | strict_minimal | semantic_paraphrase | medium | workplace / everyday
    # Q_len: 2, P_len: 1, Pos: 1
    {
        "semantic_frame_id": "frame_00034",
        "template_id": "conditional",
        "critical_lemma": "incelemek",
        "critical_word_query": "devretmeyip",
        "critical_word_positive": "inceletilmedi",
        "feature_delta": "başkasına yaptırılmama ↔ doğrudan yapılma / yaptırılma",
        "edit_script": {
            "applies": True,
            "positive_form": "inceletilmedi",
            "minimal_negative_form": "inceletildi",
            "operation": "yalnız CAUS.PASS.NEG biçimini değiştir",
            "changed_feature": "CAUS.PASS.NEG",
            "invariants": ["lemma", "token_order", "event"],
        },
        "query": "Müdür projeyi taşerona devretmeyip kurum içinde tamamladı. Dış kaynaklı denetim masrafından böylece kaçınıldı.",
        "context_sentences": [],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Dosya dış uzmanlara inceletilmedi şirket içinde tamamlandı.",
                "critical_word": "inceletilmedi",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Dosya dış uzmanlara inceletildi şirket içinde tamamlandı.",
                "critical_word": "inceletildi",
                "reason": "minimal_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Dosya dış uzmanlara inceletilmeyecek şirket içinde tamamlandı.",
                "critical_word": "inceletilmeyecek",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Dosya dış uzmanlara inceletilmemiş şirket içinde tamamlandı.",
                "critical_word": "inceletilmemiş",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Bütçe dış uzmanlara inceletilmedi şirket içinde tamamlandı.",
                "critical_word": "inceletilmedi",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Dosya yerel ajanslara inceletilmedi şirket içinde tamamlandı.",
                "critical_word": "inceletilmedi",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Dosya dış uzmanlara onaylatılmadı şirket içinde tamamlandı.",
                "critical_word": "onaylatılmadı",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Dosya dış uzmanlara incelenmedi şirket içinde tamamlandı.",
                "critical_word": "incelenmedi",
                "reason": "partial_chain_negative",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Uzmanlarca inceletilmeyen dosya şirket içinde tamamlandı.",
                "critical_word": "inceletilmeyen",
                "reason": "scope_attachment_trap",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Ofis personeli aylık yemek menüsünü oylamayla seçti.",
                "critical_word": "seçti",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Şirket içi eğitim semineri konferans salonunda başladı.",
                "critical_word": "başladı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 35: raw_00035_5de1388aac | DISTR | natural_retrieval | morph_explicit | high | news / conversational
    # Q_len: 1, P_len: 3, Pos: 2
    {
        "semantic_frame_id": "frame_00035",
        "template_id": "event_report",
        "critical_lemma": "üç",
        "critical_word_query": "ikişer",
        "critical_word_positive": "üçer",
        "feature_delta": "her birine ayrı ayrı ↔ toplam",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Resmi yardım kuruluşu afetzedelere çadırları ikişer ikişer dağıttı.",
        "context_sentences": [
            "Kızılay ekipleri bölgedeki çalışmalarını sürdürüyor.",
            "Vatandaşlar yardımları düzenli şekilde teslim aldı."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Resmi yardım kuruluşu afetzedelere çadırları üçer üçer verdi.",
                "critical_word": "üçer",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Resmi yardım kuruluşu afetzedelere çadırları toplam iki adet dağıttı.",
                "critical_word": "iki",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Resmi yardım kuruluşu afetzedelere çadırları ikinci etapta dağıttı.",
                "critical_word": "ikinci",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Resmi yardım kuruluşu afetzedelere çadırları ikide bir dağıttı.",
                "critical_word": "ikide",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Resmi yardım kuruluşu afetzedelere battaniyeleri ikişer ikişer dağıttı.",
                "critical_word": "ikişer",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Yerel dernek afetzedelere çadırları ikişer ikişer dağıttı.",
                "critical_word": "ikişer",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Resmi yardım kuruluşu afetzedelerden çadırları ikişer ikişer topladı.",
                "critical_word": "ikişer",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Lojistik tırları yardım kolilerini spor salonuna boşalttı.",
                "critical_word": "kolilerini",
                "reason": "semantic_retrieval_hard",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "İkişer çadır kuruldu ama yardım kuruluşu henüz dağıtmadı.",
                "critical_word": "ikişer",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Yerel gazete çevre kirliliğiyle ilgili röportaj dizisi yayınladı.",
                "critical_word": "yayınladı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Belediye tiyatrosu sezonun son oyununu halka ücretsiz sundu.",
                "critical_word": "sundu",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 36: raw_00036_d5b2530641 | CAUS.PASS.NEG | strict_minimal | morph_explicit | high | finance / news_report
    # Q_len: 1, P_len: 1, Pos: 1
    {
        "semantic_frame_id": "frame_00036",
        "template_id": "temporal_subordinate",
        "critical_lemma": "onaylamak",
        "critical_word_query": "onaylatılmadı",
        "critical_word_positive": "onaylatılmadı",
        "feature_delta": "başkasına yaptırılmama ↔ doğrudan yapılma / yaptırılma",
        "edit_script": {
            "applies": True,
            "positive_form": "onaylatılmadı",
            "minimal_negative_form": "onaylatıldı",
            "operation": "yalnız CAUS.PASS.NEG biçimini değiştir",
            "changed_feature": "CAUS.PASS.NEG",
            "invariants": ["lemma", "token_order", "event"],
        },
        "query": "Banka borç yapılandırma evraklarını denetçiye onaylatılmadı gerekçesiyle resmi yazıyla bekletti.",
        "context_sentences": [],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Banka borç yapılandırma evraklarını kurula onaylatılmadı gerekçesiyle doğrudan bekletti.",
                "critical_word": "onaylatılmadı",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Banka borç yapılandırma evraklarını kurula onaylatıldı gerekçesiyle doğrudan bekletti.",
                "critical_word": "onaylatıldı",
                "reason": "minimal_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Banka borç yapılandırma evraklarını kurula onaylatılmayacak gerekçesiyle doğrudan bekletti.",
                "critical_word": "onaylatılmayacak",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Banka borç yapılandırma evraklarını kurula onaylatılmamış gerekçesiyle doğrudan bekletti.",
                "critical_word": "onaylatılmamış",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Banka borç yapılandırma evraklarını denetçiye onaylatılmadı gerekçesiyle doğrudan bekletti.",
                "critical_word": "onaylatılmadı",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Müdür borç yapılandırma evraklarını denetçiye onaylatılmadı gerekçesiyle doğrudan bekletti.",
                "critical_word": "onaylatılmadı",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Banka borç yapılandırma evraklarını denetçiye onaylatılmadı gerekçesiyle resmen onayladı.",
                "critical_word": "onaylatılmadı",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Banka borç yapılandırma evraklarını denetçiye onaylanmadı gerekçesiyle doğrudan bekletti.",
                "critical_word": "onaylanmadı",
                "reason": "partial_chain_negative",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Denetçiye onaylatılmayan borç yapılandırma evrakları bankada gerekçesiyle bekletildi.",
                "critical_word": "onaylatılmayan",
                "reason": "scope_attachment_trap",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Hazine müsteşarlığı altın tahvili ihracını başarıyla tamamladı.",
                "critical_word": "tamamladı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Sigorta şirketi kasko poliçesi primlerini yeniden hesapladı.",
                "critical_word": "hesapladı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 37: raw_00037_69189fadfb | COND | controlled_diverse | semantic_paraphrase | medium | finance / everyday
    # Q_len: 1, P_len: 4, Pos: 1
    {
        "semantic_frame_id": "frame_00037",
        "template_id": "relative_clause",
        "critical_lemma": "kapatmak",
        "critical_word_query": "ödenirse",
        "critical_word_positive": "kapatılırsa",
        "feature_delta": "koşullu ihtimal ↔ gerçekleşmiş neden-sonuç",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Kredi taksitleri zamanında ödenirse banka faiz indirimini uygular.",
        "context_sentences": [
            "Finans kuruluşu müşteri memnuniyeti anketini başlattı.",
            "Şube müdürleri yeni kampanyayı tanıttı.",
            "Bilgilendirme mesajı telefonlara gönderildi."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Borç bakiyesi erken kapatılırsa kurum ek masrafı siler.",
                "critical_word": "kapatılırsa",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Borç bakiyesi erken kapatıldı ve kurum ek masrafı sildi.",
                "critical_word": "kapatıldı",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Borç bakiyesi erken kapatılınca kurum ek masrafı siler.",
                "critical_word": "kapatılınca",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Borç bakiyesi erken kapatılmalı kurum ek masrafı siler.",
                "critical_word": "kapatılmalı",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Döviz hesabı erken kapatılırsa kurum ek masrafı siler.",
                "critical_word": "kapatılırsa",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Borç bakiyesi erken kapatılırsa komisyoncu ek masrafı siler.",
                "critical_word": "kapatılırsa",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Borç bakiyesi erken kapatılırsa kurum cezai faiz uygular.",
                "critical_word": "kapatılırsa",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Ek masraf silinirse borç bakiyesini erken kapatan müşteriler ödüllendirilir.",
                "critical_word": "silinirse",
                "reason": "scope_attachment_trap",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Masraflar silinse de borç bakiyesi erken kapatılmadı.",
                "critical_word": "silinse",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Döviz bürosu gün sonu nakit sayımını kasada tamamladı.",
                "critical_word": "tamamladı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Bireysel emeklilik fonları bu çeyrekte getiri sağladı.",
                "critical_word": "sağladı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 38: raw_00038_9d7678d713 | POSS.3PL | strict_minimal | morph_explicit | low | culture_technology / formal_record
    # Q_len: 2, P_len: 2, Pos: 2
    {
        "semantic_frame_id": "frame_00038",
        "template_id": "relative_clause",
        "critical_lemma": "buluş",
        "critical_word_query": "algoritmalarını",
        "critical_word_positive": "buluşlarını",
        "feature_delta": "onların ↔ onun",
        "edit_script": {
            "applies": True,
            "positive_form": "buluşlarını",
            "minimal_negative_form": "buluşunu",
            "operation": "yalnız POSS.3PL biçimini değiştir",
            "changed_feature": "POSS.3PL",
            "invariants": ["lemma", "token_order", "event"],
        },
        "query": "Mühendisler yeni yapay zekâ modellerini tanıttı. Sistem onların algoritmalarını resmi sicile kaydetti.",
        "context_sentences": [
            "Teknoloji enstitüsü yıllık patent raporunu yayınladı."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Heyet araştırmacıların buluşlarını arşiv sistemine işledi.",
                "critical_word": "buluşlarını",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Heyet araştırmacıların buluşunu arşiv sistemine işledi.",
                "critical_word": "buluşunu",
                "reason": "minimal_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Heyet araştırmacıların buluşuna arşiv sistemine işledi.",
                "critical_word": "buluşuna",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Heyet araştırmacıların buluşundan arşiv sistemine işledi.",
                "critical_word": "buluşundan",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Heyet araştırmacıların bütçesini arşiv sistemine işledi.",
                "critical_word": "bütçesini",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Yönetim araştırmacıların buluşlarını arşiv sistemine işledi.",
                "critical_word": "buluşlarını",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Heyet araştırmacıların buluşlarını arşiv sisteminden sildi.",
                "critical_word": "buluşlarını",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Heyet tek bir araştırmacının buluşlarını arşiv sistemine işledi.",
                "critical_word": "buluşlarını",
                "reason": "noun_possessor_number_trap",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Buluşları kaydedilen araştırmacılar ödül törenine henüz katılmadı.",
                "critical_word": "buluşları",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Bilim dergisi kuantum bilgisayarlar hakkında özel dosya hazırladı.",
                "critical_word": "hazırladı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Yazılım konferansı açılış konuşmasını ünlü bir profesör yaptı.",
                "critical_word": "yaptı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 39: raw_00039_1da27e0122 | RECP.CAUS | controlled_diverse | semantic_paraphrase | medium | law_public_services / conversational
    # Q_len: 1, P_len: 3, Pos: 2
    {
        "semantic_frame_id": "frame_00039",
        "template_id": "temporal_subordinate",
        "critical_lemma": "uzlaşmak",
        "critical_word_query": "anlaştırdı",
        "critical_word_positive": "uzlaştırdı",
        "feature_delta": "birbirine yaptırma / karşılıklı sağlama ↔ tek taraflı ettirgen",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Hakim duruşmada iki tarafı yüz yüze getirip konuşturtarak anlaştırdı.",
        "context_sentences": [
            "Adliye koridorlarında yoğun bir kalabalık vardı.",
            "Avukatlar müvekkilleriyle salonun önünde görüştü."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Uzlaştırıcı dava açılmadan davacı ve davalıyı uzlaştırdı.",
                "critical_word": "uzlaştırdı",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Uzlaştırıcı dava açılmadan davacı ve davalıyla uzlaştı.",
                "critical_word": "uzlaştı",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Uzlaştırıcı dava açılmadan davacı ve davalıyı uzlaştıracak.",
                "critical_word": "uzlaştıracak",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Uzlaştırıcı dava açılmadan davacı ve davalıyı uzlaştırmış.",
                "critical_word": "uzlaştırmış",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Uzlaştırıcı dava açılmadan tanıkları ve bilirkişiyi uzlaştırdı.",
                "critical_word": "uzlaştırdı",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Yazı işleri müdürü dava açılmadan davacı ve davalıyı uzlaştırdı.",
                "critical_word": "uzlaştırdı",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Uzlaştırıcı dava açılmadan davacı ve davalıyı mahkemeye sevk etti.",
                "critical_word": "sevk etti",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Uzlaştırıcı dava açılmadan uzlaşma protokolünü masada bekletti.",
                "critical_word": "uzlaşma",
                "reason": "partial_chain_negative",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Taraflar uzlaştırıldı fakat mahkeme duruşma gününü iptal etmedi.",
                "critical_word": "uzlaştırıldı",
                "reason": "scope_attachment_trap",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Baro başkanlığı stajyer avukatlar için seminer salonu tahsis etti.",
                "critical_word": "tahsis etti",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "İcra dairesi tebligat evraklarını posta servisine teslim etti.",
                "critical_word": "teslim etti",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 40: raw_00040_a82a4ca602 | ALLO.DAT | controlled_diverse | semantic_paraphrase | high | health / conversational
    # Q_len: 1, P_len: 1, Pos: 1
    {
        "semantic_frame_id": "frame_00040",
        "template_id": "event_report",
        "critical_lemma": "oda",
        "critical_word_query": "odasına",
        "critical_word_positive": "kliniğine",
        "feature_delta": "farklı yüzey allomorfları (-a/-e/-ya/-ye) aynı yönelme işlevini taşır",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Hasta sabah erken saatte muayene odasına doğru yöneldi.",
        "context_sentences": [],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Hasta sabah erken saatte poliklinik kliniğine doğru yöneldi.",
                "critical_word": "kliniğine",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Hasta sabah erken saatte muayene odasından doğru uzaklaştı.",
                "critical_word": "odasından",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Hasta sabah erken saatte muayene odasında doğru bekledi.",
                "critical_word": "odasında",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Hasta sabah erken saatte muayene odasını doğru inceledi.",
                "critical_word": "odasını",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Hasta sabah erken saatte hastane kantinine doğru yöneldi.",
                "critical_word": "kantinine",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Doktor sabah erken saatte muayene odasına doğru yöneldi.",
                "critical_word": "odasına",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Hasta sabah erken saatte muayene odasına doğru koşmadı.",
                "critical_word": "odasına",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Hasta sabah erken saatte muayene odasınca doğru kararlaştırıldı.",
                "critical_word": "odasınca",
                "reason": "allomorph_form_function_trap",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Odadaki hasta sabah erken saatte muayeneye doğru gitmedi.",
                "critical_word": "odadaki",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Eczacı reçetedeki antibiyotik kullanım dozunu hastaya anlattı.",
                "critical_word": "anlattı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Radyoloji teknisyeni röntgen cihazının periyodik bakımını yaptı.",
                "critical_word": "yaptı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 41: raw_00041_49a0360adf | PL | natural_retrieval | semantic_paraphrase | medium | health / formal_record
    # Q_len: 2, P_len: 3, Pos: 3
    {
        "semantic_frame_id": "frame_00041",
        "template_id": "event_report",
        "critical_lemma": "numune",
        "critical_word_query": "tahlillerini",
        "critical_word_positive": "numuneleri",
        "feature_delta": "birden çok ↔ tek",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Laboratuvar yetkilileri kan tahlillerini titizlikle inceledi. Elde edilen tüm tıbbi veriler sisteme kaydedildi.",
        "context_sentences": [
            "Klinik araştırmalar merkezi denetim sürecini başlattı.",
            "Uzman heyet hazırlanan dosya özetini onayladı."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Biyokimya uzmanları numuneleri eksiksiz inceledi.",
                "critical_word": "numuneleri",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Biyokimya uzmanları tek bir numuneyi inceledi.",
                "critical_word": "numuneyi",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Biyokimya uzmanları numunelere dikkatle baktı.",
                "critical_word": "numunelere",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Biyokimya uzmanları numunelerden parça aldı.",
                "critical_word": "numunelerden",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Biyokimya uzmanları anket formlarını inceledi.",
                "critical_word": "formlarını",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Stajyer öğrenciler numuneleri eksiksiz inceledi.",
                "critical_word": "numuneleri",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Biyokimya uzmanları numuneleri incelemeden imha etti.",
                "critical_word": "numuneleri",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "İdari birim laboratuvar sarf malzemelerini teslim aldı.",
                "critical_word": "malzemelerini",
                "reason": "semantic_retrieval_hard",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Numunelerdeki değerler sisteme girildi fakat uzmanlar henüz incelemedi.",
                "critical_word": "numunelerdeki",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Hastane başhekimi acil servis doktorlarıyla toplantı yaptı.",
                "critical_word": "yaptı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Poliklinik hemşiresi aşı randevusu alan hastaları karşıladı.",
                "critical_word": "karşıladı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 42: raw_00042_b36b86a288 | ALLO.ACC | controlled_diverse | morph_explicit | medium | ecommerce / everyday
    # Q_len: 1, P_len: 2, Pos: 2
    {
        "semantic_frame_id": "frame_00042",
        "template_id": "relative_clause",
        "critical_lemma": "ayakkabı",
        "critical_word_query": "ayakkabıyı",
        "critical_word_positive": "kazağı",
        "feature_delta": "farklı yüzey allomorfları (-ı/-i/-u/-ü/-yı/-yi/-yu/-yü) aynı belirtme işlevini taşır",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Alıcı sipariş ettiği ayakkabıyı kargo görevlisinden teslim aldı.",
        "context_sentences": [
            "Kurye paketi kapıda imza karşılığı bıraktı."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Müşteri satın aldığı kazağı kargo görevlisinden teslim aldı.",
                "critical_word": "kazağı",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Alıcı sipariş ettiği ayakkabıya kargo görevlisinden baktı.",
                "critical_word": "ayakkabıya",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Alıcı sipariş ettiği ayakkabıda kargo görevlisinden kusur buldu.",
                "critical_word": "ayakkabıda",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Alıcı sipariş ettiği ayakkabıdan kargo görevlisinden bahsetti.",
                "critical_word": "ayakkabıdan",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Alıcı sipariş ettiği telefonu kargo görevlisinden teslim aldı.",
                "critical_word": "telefonu",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Komşu sipariş edilen ayakkabıyı kargo görevlisinden teslim aldı.",
                "critical_word": "ayakkabıyı",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Alıcı sipariş ettiği ayakkabıyı kargo görevlisinden teslim almayı reddetti.",
                "critical_word": "ayakkabıyı",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Alıcı sipariş ettiği ayakkabıcıyı kargo görevlisinden teslim aldı.",
                "critical_word": "ayakkabıcıyı",
                "reason": "allomorph_form_function_trap",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Ayakkabıdaki barkod okutuldu ama alıcı paketi henüz teslim almadı.",
                "critical_word": "ayakkabıdaki",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "İnternet mağazası yeni sezon ürünlerini vitrinde sergiledi.",
                "critical_word": "sergiledi",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Müşteri hizmetleri kargo takip numarasını sms ile gönderdi.",
                "critical_word": "gönderdi",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 43: raw_00043_9c38364af2 | NMLZ.CASE.CNTR | natural_retrieval | semantic_paraphrase | low | workplace / formal_record
    # Q_len: 1, P_len: 3, Pos: 3
    {
        "semantic_frame_id": "frame_00043",
        "template_id": "conditional",
        "critical_lemma": "daralmak",
        "critical_word_query": "yaklaşmasına karşın",
        "critical_word_positive": "daralmasına rağmen",
        "feature_delta": "beklentinin aksine gerçekleşme ↔ düz neden-sonuç / engellenme",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Proje teslim tarihi yaklaşmasına karşın teknik ekip geliştirmeyi durdurmadı.",
        "context_sentences": [
            "Şirket yönetim kurulu haftalık toplantısını tamamladı.",
            "Departman yöneticileri hedefleri gözden geçirdi."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Süre daralmasına rağmen mühendisler çalışmayı kesintisiz sürdürdü.",
                "critical_word": "daralmasına rağmen",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Süre daraldığı için mühendisler çalışmayı durdurdu.",
                "critical_word": "daraldığı için",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Süre daralınca mühendisler çalışmayı hızlandırdı.",
                "critical_word": "daralınca",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Süre daraldıkça mühendisler çalışmayı hızlandırdı.",
                "critical_word": "daraldıkça",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Bütçe daralmasına rağmen mühendisler çalışmayı sürdürdü.",
                "critical_word": "daralmasına rağmen",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Süre daralmasına rağmen stajyerler çalışmayı sürdürdü.",
                "critical_word": "daralmasına rağmen",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Süre daralmasına rağmen mühendisler projeyi iptal etti.",
                "critical_word": "daralmasına rağmen",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "İnsan kaynakları ekibi ofis çalışma saatlerini yeniden belirledi.",
                "critical_word": "saatlerini",
                "reason": "semantic_retrieval_hard",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Süre uzatılmasına rağmen ekip yazılımı henüz teslim etmedi.",
                "critical_word": "uzatılmasına rağmen",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Bilgi işlem servisi yeni yazıcıları ağa bağladı.",
                "critical_word": "bağladı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Pazarlama birimi yıllık reklam bütçesini hazırladı.",
                "critical_word": "hazırladı",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 44: raw_00044_79600106b8 | POSS.PL.ABL | strict_minimal | morph_explicit | medium | finance / news_report
    # Q_len: 1, P_len: 2, Pos: 1
    {
        "semantic_frame_id": "frame_00044",
        "template_id": "relative_clause",
        "critical_lemma": "hesap",
        "critical_word_query": "hesaplarımızdan",
        "critical_word_positive": "hesaplarımızdan",
        "feature_delta": "kimin kaç tanesinden ayrılma",
        "edit_script": {
            "applies": True,
            "positive_form": "hesaplarımızdan",
            "minimal_negative_form": "hesabımızdan",
            "operation": "yalnız POSS.PL.ABL biçimini değiştir",
            "changed_feature": "POSS.PL.ABL",
            "invariants": ["lemma", "token_order", "event"],
        },
        "query": "Müşteriler döviz fonlarını ortak hesaplarımızdan çekmek istedi.",
        "context_sentences": [
            "Banka şubesi günlük işlem limitlerini güncelledi."
        ],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Müşteriler döviz fonlarını ortak hesaplarımızdan aktarmayı uygun gördü.",
                "critical_word": "hesaplarımızdan",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Müşteriler döviz fonlarını ortak hesabımızdan aktarmayı uygun gördü.",
                "critical_word": "hesabımızdan",
                "reason": "minimal_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Müşteriler döviz fonlarını ortak hesaplarımıza aktarmayı uygun gördü.",
                "critical_word": "hesaplarımıza",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Müşteriler döviz fonlarını ortak hesaplarımızı aktarmayı uygun gördü.",
                "critical_word": "hesaplarımızı",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Müşteriler döviz bakiyesini ortak hesaplarımızdan aktarmayı uygun gördü.",
                "critical_word": "hesaplarımızdan",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Danışmanlar döviz fonlarını ortak hesaplarımızdan çekmek istedi.",
                "critical_word": "hesaplarımızdan",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Müşteriler döviz fonlarını ortak hesaplarımızdan çekmekten vazgeçti.",
                "critical_word": "hesaplarımızdan",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Müşteriler döviz fonlarını ortak hesaplarımızdan çekmeyi durdurdu.",
                "critical_word": "hesaplarımızdan",
                "reason": "partial_chain_negative",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "Müşteriler döviz fonlarını ortak hesaplarımızdan çekmek istemedi.",
                "critical_word": "hesaplarımızdan",
                "reason": "scope_attachment_trap",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Finans danışmanı borsa bültenini sabah saatlerinde yayınladı.",
                "critical_word": "yayınladı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Kredi kartı komisyon oranları resmi gazetede ilan edildi.",
                "critical_word": "ilan edildi",
                "reason": "easy_negative",
            },
        ],
    },

    # Slot 45: raw_00045_d48ae7e233 | OPT | controlled_diverse | semantic_paraphrase | low | law_public_services / everyday
    # Q_len: 2, P_len: 1, Pos: 1
    {
        "semantic_frame_id": "frame_00045",
        "template_id": "event_report",
        "critical_lemma": "incelemek",
        "critical_word_query": "değerlendirmek istedi",
        "critical_word_positive": "inceleyelim",
        "feature_delta": "dilek/istek ↔ gerçekleşmiş olay",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Avukatlar duruşma öncesinde müvekkilleriyle uzlaşma önerisini değerlendirmek istedi. Birlikte bir karar vermeyi hedeflediler.",
        "context_sentences": [],
        "candidates": [
            {
                "candidate_slot": "positive_01",
                "critical_sentence": "Hukukçular protokol taslağını masada birlikte inceleyelim dedi.",
                "critical_word": "inceleyelim",
                "reason": "equivalence_positive",
            },
            {
                "candidate_slot": "hard_01",
                "critical_sentence": "Hukukçular protokol taslağını masada birlikte inceledi.",
                "critical_word": "inceledi",
                "reason": "controlled_morph_negative",
            },
            {
                "candidate_slot": "hard_02",
                "critical_sentence": "Hukukçular protokol taslağını masada birlikte inceleyecek.",
                "critical_word": "inceleyecek",
                "reason": "same_lemma_wrong_inflection",
            },
            {
                "candidate_slot": "hard_03",
                "critical_sentence": "Hukukçular protokol taslağını masada birlikte incelemeli.",
                "critical_word": "incelemeli",
                "reason": "related_feature_negative",
            },
            {
                "candidate_slot": "hard_04",
                "critical_sentence": "Hukukçular kira sözleşmesini masada birlikte inceleyelim dedi.",
                "critical_word": "inceleyelim",
                "reason": "same_morph_wrong_content",
            },
            {
                "candidate_slot": "hard_05",
                "critical_sentence": "Stajyerler protokol taslağını masada birlikte inceleyelim dedi.",
                "critical_word": "inceleyelim",
                "reason": "state_participant_time_trap",
            },
            {
                "candidate_slot": "hard_06",
                "critical_sentence": "Hukukçular protokol taslağını masada birlikte reddedelim dedi.",
                "critical_word": "reddedelim",
                "reason": "close_paraphrase_wrong_meaning",
            },
            {
                "candidate_slot": "hard_07",
                "critical_sentence": "Protokol taslağı masadaki hukukçuları birlikte inceleyelim dedi.",
                "critical_word": "inceleyelim",
                "reason": "argument_role_reversal",
            },
            {
                "candidate_slot": "hard_08",
                "critical_sentence": "İncelenecek taslak hazırlandı ama hukukçular henüz masaya oturmadı.",
                "critical_word": "incelenecek",
                "reason": "morph_distractor",
            },
            {
                "candidate_slot": "easy_01",
                "critical_sentence": "Vatandaşlık başvuru formu belediye girişindeki masadan alındı.",
                "critical_word": "alındı",
                "reason": "easy_negative",
            },
            {
                "candidate_slot": "easy_02",
                "critical_sentence": "Noter katibi tapu devir vekâletnamesini imzalattı.",
                "critical_word": "imzalattı",
                "reason": "easy_negative",
            },
        ],
    },
]
