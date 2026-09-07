# POZZ — UX/UI decisions

Zapis stanu prac na ekranach. Mocki HTML mogą dojść później w `screens/` — do tego czasu spec zachowania w `ux-ui-spec.md` jest SoT.

## Kierunek wizualny

Design system **Classical** (jak Langy, kliniczny): serif nagłówków, hairlines, złoto tylko jako obrys. Dark: #16130f / #1e1a15. **Bez emoji w chrome** (ikony jednej rodziny). Copy UI po polsku.

Taste overlay: VARIANCE 3 / MOTION 2 / DENSITY 6. `prefers-reduced-motion` obowiązkowe.

Classical jest SoT wizualnym. Nie dodawać równoległych rules o kolorach w `.cursor/rules/`.

## IA

Dolny pasek: **Symulacja / Wywiad / Menu**. Menu zawiera **Sesje** (historia). Admin tylko dla `is_admin`.

## Ekrany (kanwa)

Kanwa 390 × 844 (mobile), admin 1180 × 760. Hit targets ≥ 44 px.

| Ekran | Zachowanie |
|---|---|
| Login / waiting | OAuth; wait aż Accept |
| Symulacja | karta + transkrypt + composer + tryby |
| Zakończenie | plan leczenia + ocena |
| Wywiad nagranie | Start/Stop + wynik |
| Wywiad ręczny | linie + podsumowanie |
| Sesje | lista + szczegół |
| Admin | users / cap / generate |

## Ton copy

Polski UI. Błędy = co się stało + co zrobić. Puste stany afirmatywne. Disclaimer treningowy, bez tonu „diagnoza”.

## Do domknięcia (UX mocki)

- HTML screens w `screens/` (opcjonalnie, jak Langy `.dc.html`)
- Desktop layout karta + czat
- PWA: mic prompt, A2HS, offline shell
- Tokeny CSS zamiast hardcode Classical
