# Kvantová a klasická částice v jámě — Streamlit app

Jednoduchá výuková Streamlit aplikace pro studenty fyziky:

- nekonečně hluboká 1D kvantová jáma,
- časový vývoj vlnového balíku a srovnání s klasickou částicí,
- dvojitá kvantová jáma a tunelování,
- optická analogie se dvěma vlnovody,
- konečně hluboká kvantová jáma,
- vybrané 3D pohledy.

## Lokální spuštění

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktura repozitáře

```text
.
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Nasazení na Streamlit Community Cloud

1. Nahraj tento obsah do nového GitHub repozitáře.
2. Na Streamlit Community Cloud zvol **New app**.
3. Vyber svůj GitHub repozitář.
4. Jako **Main file path** nastav `app.py`.
5. Potvrď nasazení.

## Poznámky

- Závislosti jsou v `requirements.txt`.
- Není potřeba `packages.txt`, protože aplikace nepoužívá systémové balíčky instalované přes `apt`.
- Po commitu nových změn do GitHub repozitáře se nasazená aplikace automaticky aktualizuje.
