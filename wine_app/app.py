import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def load_default_threshold():
    return json.loads((APP_DIR / 'config.json').read_text())['threshold']


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
test = load_test()
FEATURES = list(test.columns.drop('target'))
X_test = test[FEATURES]
y_test = test['target'].to_numpy()
proba_test = test_proba(model, X_test)

st.title('Подборка вин')

main, side = st.columns([4, 1])

with side:
    st.header('Порог модели')
    threshold = st.slider('Порог t', 0.0, 1.0, float(load_default_threshold()), 0.01)
    st.write('Вино попадает в подборку, если вероятность хорошего вина не ниже порога.')

tab1, tab2, tab3, tab4 = main.tabs(['Оценка вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with tab1:
    st.subheader('Оценка одного вина')
    values = {}
    cols = st.columns(3)
    i = 0
    for feature in FEATURES:
        if feature == 'is_red':
            continue
        col = cols[i % 3]
        i += 1
        lo, hi, med = float(X_test[feature].min()), float(X_test[feature].max()), float(X_test[feature].median())
        values[feature] = col.number_input(feature, min_value=lo, max_value=hi, value=med, step=(hi - lo) / 100,
                                           format='%.4f')
    wine_type = st.radio('Тип вина', ['белое', 'красное'], horizontal=True)
    values['is_red'] = 1 if wine_type == 'красное' else 0

    wine = pd.DataFrame([values])[FEATURES]
    p = model.predict_proba(wine)[0, 1]
    st.metric('Вероятность хорошего вина', f'{p:.3f}')
    if p >= threshold:
        st.success(f'Решение модели: хорошее вино (порог {threshold:.2f})')
    else:
        st.warning(f'Решение модели: не хорошее вино (порог {threshold:.2f})')

with tab2:
    st.subheader('Метрики на тестовой выборке')
    y_pred = (proba_test >= threshold).astype(int)

    cm = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric('Precision', f'{precision:.3f}')
    c2.metric('Recall', f'{recall:.3f}')
    c3.metric('F1', f'{f1:.3f}')

    fig, ax = plt.subplots(figsize=(4, 3.5))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['не хорошее', 'хорошее']).plot(ax=ax, colorbar=False)
    ax.set_title('Матрица ошибок')
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, proba_test)
    prec_curve, rec_curve, _ = metrics.precision_recall_curve(y_test, proba_test)
    cur_fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    cur_tpr = recall

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(fpr, tpr)
    axs[0].plot([0, 1], [0, 1], '--', color='gray')
    axs[0].scatter(cur_fpr, cur_tpr, color='red', zorder=3, label=f'порог {threshold:.2f}')
    axs[0].set_title('ROC-кривая')
    axs[0].set_xlabel('FPR')
    axs[0].set_ylabel('TPR')
    axs[0].legend()

    axs[1].plot(rec_curve, prec_curve)
    axs[1].axhline(y_test.mean(), linestyle='--', color='gray')
    axs[1].scatter(recall, precision, color='red', zorder=3, label=f'порог {threshold:.2f}')
    axs[1].set_title('PR-кривая')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        'Если порог увеличить, в подборку будет попадать меньше вин, но среди них будет больше действительно '
        'хороших (precision растет), зато мы будем чаще пропускать хорошие вина (recall падает). '
        'Если порог уменьшить, подборка станет шире и пропусков хороших вин будет меньше (recall растет), '
        'но в нее будет попадать больше посредственных вин (precision падает).'
    )

with tab3:
    st.subheader('Пакетное предсказание')
    st.write('Загрузите CSV-файл с признаками вин (без столбца `target`). Порог берется из сайдбара.')
    file = st.file_uploader('CSV-файл', type='csv')
    if file is not None:
        try:
            batch = pd.read_csv(file)
        except Exception:
            st.error('Не удалось прочитать файл как CSV.')
            batch = None

        if batch is not None:
            missing = [f for f in FEATURES if f not in batch.columns]
            if missing:
                st.error(f'В файле не хватает столбцов: {", ".join(missing)}')
            else:
                X_batch = batch[FEATURES].apply(pd.to_numeric, errors='coerce')
                if X_batch.isna().any().any():
                    st.error('В признаках есть пропуски или нечисловые значения.')
                else:
                    result = batch.copy()
                    result['proba'] = model.predict_proba(X_batch)[:, 1]
                    result['decision'] = (result['proba'] >= threshold).astype(int)
                    st.dataframe(result)
                    st.download_button('Скачать результат', result.to_csv(index=False).encode('utf-8'),
                                       file_name='predictions.csv', mime='text/csv')

with tab4:
    st.subheader('Коэффициенты логистической регрессии')
    st.write('Признаки масштабируются, поэтому коэффициенты можно сравнивать между собой.')
    coef = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coef.index, coef.values, color=np.where(coef.values > 0, 'tab:green', 'tab:red'))
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Коэффициент')
    st.pyplot(fig)
    plt.close(fig)

    top_for = ', '.join(coef.sort_values(ascending=False).index[:3])
    top_against = ', '.join(coef.index[:3])
    st.write(f'Сильнее всего «за» хорошее вино (зеленые): {top_for}.')
    st.write(f'Сильнее всего «против» хорошего вина (красные): {top_against}.')
