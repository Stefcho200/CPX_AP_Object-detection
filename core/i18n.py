"""Minimal in-house i18n system. Add new languages by adding a new
dict below and registering it in LANGUAGES. Add new strings by adding
a new key to EVERY language dict (English is the source of truth -
run check_missing_keys() during dev to catch gaps).

Usage in GUI code:
    from core.i18n import t
    tk.Label(self, text=t("training_menu_title"))

To add a language later: copy the EN block, translate every value,
add it to LANGUAGES with its flag emoji + display name.
"""

CURRENT_LANG = "en"

_listeners = []  # callables invoked when language changes, so open
                  # windows can refresh their text live if desired


def on_language_change(callback):
    _listeners.append(callback)


def set_language(lang_code):
    global CURRENT_LANG
    if lang_code not in LANGUAGES:
        return
    CURRENT_LANG = lang_code
    for cb in list(_listeners):
        try:
            cb(lang_code)
        except Exception:
            pass


def get_language():
    return CURRENT_LANG


def t(key):
    """Translates `key` into the current language. Falls back to
    English, then to the key itself, if missing - so a missing
    translation never crashes the app, it just shows English or the
    raw key (easy to spot and fix)."""
    lang_dict = LANGUAGES.get(CURRENT_LANG, {}).get("strings", {})
    if key in lang_dict:
        return lang_dict[key]
    en_dict = LANGUAGES["en"]["strings"]
    return en_dict.get(key, key)


def check_missing_keys():
    """Dev utility: returns {lang_code: [missing_keys]} - run manually
    while adding new strings/languages."""
    en_keys = set(LANGUAGES["en"]["strings"].keys())
    missing = {}
    for code, data in LANGUAGES.items():
        if code == "en":
            continue
        lang_keys = set(data["strings"].keys())
        gap = en_keys - lang_keys
        if gap:
            missing[code] = sorted(gap)
    return missing


# ----------------------------------------------------------------------
# English (source of truth - every key must exist here)
# ----------------------------------------------------------------------

EN_STRINGS = {
    "app_title": "CPX Product Recognizer",

    # Main menu
    "menu_training": "Training",
    "menu_detect": "Detect Products",
    "menu_settings": "Settings",

    # Training menu
    "training_title": "TRAINING",
    "train_model_on_product": "Train Model on Product",
    "training_data_cleaning": "Training Data Cleaning",
    "run_training": "Run training (build/update model)",

    "train_new_product": "Train model on new product",
    "upgrade_existing": "Upgrade model of existing product",
    "record_manual_label": "Record & manually label data",
    "resume_batch": "Resume interrupted batch",
    "relabel_session": "Relabel a recorded session",
    "delete_data": "Delete data",
    "delete_session": "Delete a recorded session",
    "delete_product": "Delete entire product",

    "distribution": "Distribution",
    "prune_frames": "Prune redundant frames from a session",
    "relabel_newest_model": "Relabel with newest model",
    "review_remove": "Review data and remove manually",
    "view_training_data": "View training data",
    "review_staged": "⚠ Review pending staged re-labels",

    "close": "Close",
    "confirm": "Confirm",
    "cancel": "Cancel",

    # Detect window
    "detect_title": "Detect Products",
    "hold_product_prompt": "Hold product close to camera to identify...",
    "tracking_status": "Tracking",
    "identified_status": "Identified",
    "line_of_sight_lost": "line of sight briefly lost...",
    "no_model_loaded": "No model loaded — showing raw camera feed",
    "ocr_confirmed": "OCR-confirmed",

    # Distribution window
    "distribution_title": "TRAINING DATA DISTRIBUTION",
    "distribution_legend": "Same color = twin product group (visually near-identical). "
                           "Click a twin group's bar to expand/collapse its breakdown.",
    "record_more_data": "📸 Record more data (weakest first)",
    "refresh": "Refresh",
    "weakest_label": "Weakest 3 groups/products (least data)",

    # Settings
    "settings_title": "Settings",
    "language_label": "Language",
    "theme_label": "Theme",
    "dark_mode": "Dark mode",
    "light_mode": "Light mode",


    # Settings Menu
    "settings_camera_title": "Camera Selection",
    "scanning_cameras": "Scanning cameras...",
    "found_camera_slots": "Found {n} camera slot(s) to check.",
    "use_auto_detect": "Use Auto-Detect (recommended)",
    "current_setting_auto": "Current setting: Auto-detect",
    "current_setting_index": "Current setting: Camera index {i}",
    "no_signal": "No signal",
    "camera_index_label": "Camera index {i}",
    "usable": "✓ Usable",
    "unusable": "✗ Black / unavailable",
    "select_btn": "Select",
    "camera_selected": "Camera {i} selected and saved.",
    "reverted_auto": "Reverted to auto-detect mode.",

    #DETECT Window Menu
    "detect_window_title": "Detect Products",
    "no_model_feed": "No model loaded — showing raw camera feed\n({error})",
    "hold_product_prompt": "Hold product close to camera to identify...",
    "tracking_status": "Tracking",
    "identified_status": "Identified",
    "tracking_lost": "Tracking {label} (line of sight briefly lost...)",
    "no_product_detected": "No product detected",
    "ocr_confirmed_suffix": " (OCR-confirmed)",


# add to EN_STRINGS
    "training_window_title": "Training",
    "training_data_title": "Training Data Distribution",

    "train_submenu_title": "TRAIN MODEL ON PRODUCT",
    "delete_submenu_title": "DELETE DATA",
    "cleaning_submenu_title": "TRAINING DATA CLEANING",

    "new_product_prompt_title": "New product",
    "new_product_prompt_text": "Enter product name (e.g. CPX11):",

    "no_data_title": "No data",
    "no_data_products": "No products recorded yet.",
    "no_data_sessions": "No sessions in this product.",
    "no_data_frames": "No frames found for this selection.",
    "empty_title": "Empty",
    "nothing_pending_title": "Nothing pending",
    "nothing_pending_batches": "No interrupted batches found.",
    "nothing_staged_title": "Nothing staged",
    "nothing_staged_text": "No staged re-label results found.",

    "select_product_upgrade": "Select product to upgrade",
    "select_product_record": "Select product to record more angles for",
    "select_product_current": "Select current product",
    "select_session": "Select session",
    "retag_to_prompt": "Re-tag '{session}' to:",
    "retag_done_title": "Done",
    "retag_done_text": "Session moved to {product}.",

    "select_scope_prune": "Select scope to prune",
    "all_products_sessions": "⚡ All products & all sessions",
    "all_sessions_of": "⚡ All sessions of {product}",
    "select_session_prune": "Select session to prune",
    "pruning_title": "Prune '{session}'",
    "similarity_dedupe_desc": "Similarity dedupe: removes frames that are\nnear-identical to the previous KEPT frame\n(adapts to how much the product actually moved).\nRecommended over decimation.",
    "similarity_threshold_label": "Similarity threshold:",
    "run_similarity_dedupe": "Run similarity dedupe",
    "run_similarity_dedupe_all": "Run similarity dedupe on ALL",
    "decimate_desc": "\nDecimate: keeps only every Nth frame\n(simple, fast, ignores content).",
    "keep_every_nth": "Keep every Nth frame:",
    "run_decimation": "Run decimation",
    "run_decimation_all": "Run decimation on ALL",
    "confirm_pruning_title": "Confirm pruning",
    "confirm_pruning_text": "This will permanently delete redundant frames from:\n\n{session}\n\nMode: {mode}\n\nThis cannot be undone. Continue?",
    "confirm_pruning_all_title": "Confirm pruning ALL sessions",
    "confirm_pruning_all_text": "This will permanently delete redundant frames from EVERY\nsession of EVERY product ({n} product(s)).\n\nMode: {mode}\n\nThis cannot be undone. Continue?",
    "pruning_complete_title": "Pruning complete",
    "pruning_complete_text": "Session: {session}\nKept: {kept} frames\nRemoved: {removed} frames",
    "pruning_complete_all_text": "Total kept: {kept}\nTotal removed: {removed}\n\nPer product:\n{summary}",
    "prune_all_title": "Prune ALL products & sessions",
    "prune_all_summary": "Pruning ALL {n} product(s),\n{total} session(s) total.",

    "relabel_window_title": "Re-label training data",
    "select_scope_relabel": "Select scope to re-label using the current best model",
    "relabel_session_title": "Re-label — {product}",
    "confirm_relabel_title": "Confirm re-labeling",
    "confirm_relabel_text": "This will run the CURRENT BEST model over {n} frame(s) and write fresh mask predictions to a STAGING area for review.\n\nYour real dataset labels will NOT be touched until you explicitly review and commit the results afterward.\n\nClass identity (which product folder) is not changed — only the mask/box geometry is re-predicted.\n\nContinue?",
    "no_model_title": "No model",
    "no_model_text": "Could not load a model: {error}",
    "relabel_in_progress_title": "Re-labeling in progress",
    "starting": "Starting...",
    "processing_frames": "Processed {done}/{total} frames...",
    "relabel_done_text": "Done. Re-labeled: {relabeled}  |  Flagged (no detection): {flagged}\n\nNothing has been saved yet — review the results next.",
    "review_staged_now": "Review staged results now",

    "select_scope_view": "Select scope to view",
    "view_title": "View — {product}",
    "viewer_all_title": "Training Data Viewer — {scope}",
    "all_products_label": "All products",

    "confirm_deletion_title": "Confirm deletion",
    "confirm_delete_session_text": "Permanently delete session:\n\n{session}\n\nThis cannot be undone.",
    "deleted_title": "Deleted",
    "deleted_session_text": "Session '{session}' has been deleted.",
    "select_product_delete": "Select product to delete",
    "confirm_delete_product_text": "Permanently delete ALL sessions for product:\n\n{product}\n\nThis cannot be undone.",
    "deleted_product_text": "Product '{product}' has been deleted.",

    "resume_batch_title": "Resume interrupted batch",
    "resume_batch_desc": "Found unfinished batch(es) — likely from a crash or closed window:",

    "training_in_progress_title": "Training in progress",
    "training_yolo": "Training YOLO detector...",
    "epoch_status": "Epoch {epoch}/{total}\nbox loss: {box_loss:.4f} | cls loss: {cls_loss:.4f}\nmAP50: {map50}",
    "training_still_running_title": "Training in progress",
    "training_still_running_text": "Training is still running in the background.\nClosing this window will NOT stop it — it will keep training until it finishes, but you won't see live progress.",


}

BG_STRINGS = {
    "app_title": "CPX Компютърно разпознаване",

    # Main menu
    "menu_training": "Обучение",
    "menu_detect": "Разпознаване на продукти",
    "menu_settings": "Настройки",

    # Training menu
    "training_title": "ОБУЧЕНИЕ",
    "train_model_on_product": "Обучи модел на продукт",
    "training_data_cleaning": "Пречистване на данни",
    "run_training": "Стартирай обучение (изгради/обнови модела)",

    "train_new_product": "Събиране на данни за нов продукт",
    "upgrade_existing": "Събиране на данни за съществуващ продукт",
    "record_manual_label": "Запис и ръчно етикетиране на данни",
    "resume_batch": "Възобнови прекъсната партида",
    "relabel_session": "Преетикетирай записана сесия",
    "delete_data": "Изтрий данни",
    "delete_session": "Изтрий записана сесия",
    "delete_product": "Изтрий целия продукт",

    "distribution": "Разпределение на данни",
    "prune_frames": "Премахни излишни кадри от сесия",
    "relabel_newest_model": "Преетикетирай с най-новия модел",
    "review_remove": "Прегледай и премахни ръчно",
    "view_training_data": "Прегледай тренировъчните данни",
    "review_staged": "⚠ Прегледай чакащи преетикетирания",

    "close": "Затвори",
    "confirm": "Потвърди",
    "cancel": "Отказ",

    # Detect window
    "detect_title": "Разпознаване на продукти",
    "hold_product_prompt": "Доближете продукта до камерата за идентификация...",
    "tracking_status": "Следене",
    "identified_status": "Идентифициран",
    "line_of_sight_lost": "връзката временно е загубена...",
    "no_model_loaded": "Няма зареден модел",
    "ocr_confirmed": "потвърдено чрез OCR",

    # Distribution window
    "distribution_title": "РАЗПРЕДЕЛЕНИЕ НА ТРЕНИРОВЪЧНИТЕ ДАННИ",
    "distribution_legend": "Еднакъв цвят = група на продукт с двойници. "
                           "Кликнете върху лентата на групата за да я разгънете/свиете.",
    "record_more_data": "Запиши още данни - най-слаби",
    "refresh": "Обнови",
    "weakest_label": "Трите най-слаби групи/продукти (най-малко данни)",

    # Settings
    "settings_title": "Настройки",
    "language_label": "Език",
    "theme_label": "Тема",
    "dark_mode": "Тъмен режим",
    "light_mode": "Светъл режим",


    #Settings Menu
    "settings_camera_title": "Избор на камера",
    "scanning_cameras": "Сканиране на камери...",
    "found_camera_slots": "Намерени {n} слот(а) за камера.",
    "use_auto_detect": "Използвай автоматично откриване (препоръчително)",
    "current_setting_auto": "Текуща настройка: Автоматично откриване",
    "current_setting_index": "Текуща настройка: Камера с индекс {i}",
    "no_signal": "Няма сигнал",
    "camera_index_label": "Камера с индекс {i}",
    "usable": "Използваема",
    "unusable": "Черна / недостъпна",
    "select_btn": "Избери",
    "camera_selected": "Камера {i} избрана и запазена.",
    "reverted_auto": "Върнато към автоматично откриване.",

    #DETECT Window Menu
    "detect_window_title": "Разпознаване на продукти",
    "no_model_feed": "Няма зареден модел — показва се суров видеопоток\n({error})",
    "hold_product_prompt": "Доближете продукта до камерата за идентификация...",
    "tracking_status": "Следене",
    "identified_status": "Идентифициран",
    "tracking_lost": "Следене на {label} (връзката временно е загубена...)",
    "no_product_detected": "Не е открит продукт",
    "ocr_confirmed_suffix": " (потвърдено чрез OCR)",


# add to BG_STRINGS
    "training_window_title": "Обучение",
    "training_data_title": "Разпределение на тренировъчните данни",

    "train_submenu_title": "ОБУЧИ МОДЕЛ НА ПРОДУКТ",
    "delete_submenu_title": "ИЗТРИЙ ДАННИ",
    "cleaning_submenu_title": "ПОЧИСТВАНЕ НА ТРЕНИРОВЪЧНИ ДАННИ",

    "new_product_prompt_title": "Нов продукт",
    "new_product_prompt_text": "Въведете име на продукт (напр. CPX-AP-I-4DI-M8-3P):",

    "no_data_title": "Няма данни",
    "no_data_products": "Все още няма записани продукти.",
    "no_data_sessions": "Няма сесии за този продукт.",
    "no_data_frames": "Няма намерени кадри за този избор.",
    "empty_title": "Празно",
    "nothing_pending_title": "Няма чакащи",
    "nothing_pending_batches": "Не са намерени прекъснати партиди.",
    "nothing_staged_title": "Няма чакащи",
    "nothing_staged_text": "Не са намерени резултати от преетикетиране.",

    "select_product_upgrade": "Изберете продукт за подобряване",
    "select_product_record": "Изберете продукт за запис на още ъгли",
    "select_product_current": "Изберете текущ продукт",
    "select_session": "Изберете сесия",
    "retag_to_prompt": "Преетикетирай '{session}' към:",
    "retag_done_title": "Готово",
    "retag_done_text": "Сесията беше преместена в {product}.",

    "select_scope_prune": "Изберете обхват за изчистване",
    "all_products_sessions": " Всички продукти и сесии",
    "all_sessions_of": " Всички сесии на {product}",
    "select_session_prune": "Изберете сесия за изчистване",
    "pruning_title": "Изчистване на '{session}'",
    "similarity_dedupe_desc": "Премахване по сходство: премахва кадри, които са\nпочти еднакви с последния ЗАПАЗЕН кадър\n(съобразява се с реалното движение на продукта).",
    "similarity_threshold_label": "Праг на сходство:",
    "run_similarity_dedupe": "Стартирай изчистване по сходство",
    "run_similarity_dedupe_all": "Стартирай изчистване по сходство на ВСИЧКИ",
    "decimate_desc": "\nДецимация: запазва само всеки N-ти кадър\n(просто, бързо, без анализ на съдържанието).",
    "keep_every_nth": "Запази всеки N-ти кадър:",
    "run_decimation": "Стартирай децимация",
    "run_decimation_all": "Стартирай децимация на ВСИЧКИ",
    "confirm_pruning_title": "Потвърди изчистването",
    "confirm_pruning_text": "Това ще изтрие завинаги излишните кадри от:\n\n{session}\n\nРежим: {mode}\n\nТова е необратимо. Продължи?",
    "confirm_pruning_all_title": "Потвърди изчистването на ВСИЧКИ сесии",
    "confirm_pruning_all_text": "Това ще изтрие завинаги излишните кадри от ВСЯКА\nсесия на ВСЕКИ продукт ({n} продукт(а)).\n\nРежим: {mode}\n\nТова е необратимо. Продължи?",
    "pruning_complete_title": "Изчистването завърши",
    "pruning_complete_text": "Сесия: {session}\nЗапазени: {kept} кадри\nПремахнати: {removed} кадри",
    "pruning_complete_all_text": "Общо запазени: {kept}\nОбщо премахнати: {removed}\n\nПо продукт:\n{summary}",
    "prune_all_title": "Изчисти ВСИЧКИ продукти и сесии",
    "prune_all_summary": "Изчистване на ВСИЧКИ {n} продукт(а),\n{total} сесии общо.",

    "relabel_window_title": "Преетикетиране на тренировъчни данни",
    "select_scope_relabel": "Изберете обхват за преетикетиране с най-новия модел",
    "relabel_session_title": "Преетикетиране — {product}",
    "confirm_relabel_title": "Потвърди преетикетирането",
    "confirm_relabel_text": "Това ще стартира НАЙ-ДОБРИЯ ТЕКУЩ модел върху {n} кадър(и) и ще запише нови предположения за маски в зона за преглед.\n\nВашите реални етикети НЕ ще бъдат променени докато изрично не прегледате и потвърдите резултатите.\n\nИдентичността на класа (папката на продукта) не се променя — само геометрията на маската/кутията се преизчислява.\n\nПродължи?",
    "no_model_title": "Няма модел",
    "no_model_text": "Моделът не можа да бъде зареден: {error}",
    "relabel_in_progress_title": "Преетикетиране в процес",
    "starting": "Стартиране...",
    "processing_frames": "Обработени {done}/{total} кадри...",
    "relabel_done_text": "Готово. Преетикетирани: {relabeled}  |  Отбелязани (без откриване): {flagged}\n\nНищо не е запазено все още — прегледайте резултатите.",
    "review_staged_now": "Прегледай резултатите сега",

    "select_scope_view": "Изберете обхват за преглед",
    "view_title": "Преглед — {product}",
    "viewer_all_title": "Преглед на тренировъчни данни — {scope}",
    "all_products_label": "Всички продукти",

    "confirm_deletion_title": "Потвърди изтриването",
    "confirm_delete_session_text": "Изтрий завинаги сесията:\n\n{session}\n\nТова е необратимо.",
    "deleted_title": "Изтрито",
    "deleted_session_text": "Сесията '{session}' беше изтрита.",
    "select_product_delete": "Изберете продукт за изтриване",
    "confirm_delete_product_text": "Изтрий завинаги ВСИЧКИ сесии на продукта:\n\n{product}\n\nТова е необратимо.",
    "deleted_product_text": "Продуктът '{product}' беше изтрит.",

    "resume_batch_title": "Възобнови прекъсната партида",
    "resume_batch_desc": "Намерени незавършени партиди — вероятно от срив или затворен прозорец:",

    "training_in_progress_title": "Обучението е в процес",
    "training_yolo": "Обучение на YOLO детектор...",
    "epoch_status": "Епоха {epoch}/{total}\nзагуба на кутия: {box_loss:.4f} | загуба на клас: {cls_loss:.4f}\nmAP50: {map50}",
    "training_still_running_title": "Обучението е в процес",
    "training_still_running_text": "Обучението продължава във фонов режим.\nЗатварянето на този прозорец НЕ ще го спре — то ще продължи докато завърши, но няма да виждате прогреса на живо.",



}

LANGUAGES = {
    "en": {"display_name": "English", "flag": "EN", "strings": EN_STRINGS},
    "bg": {"display_name": "Български", "flag": "BG", "strings": BG_STRINGS},
}

def tf(key, **kwargs):
    """Like t(), but supports .format(**kwargs) substitution for
    strings with placeholders (e.g. 'Camera {i} selected')."""
    template = t(key)
    try:
        return template.format(**kwargs)
    except Exception:
        return template


