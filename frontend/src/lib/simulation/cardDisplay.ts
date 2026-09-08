export type PatientCard = {
  name: string;
  age: string;
  has_history_here: boolean | null;
  chronic_diseases: string;
  operations: string;
  allergies: string;
  family_history: string;
};

export type CardRow = { label: string; value: string };

export function cardRowsForDisplay(card: PatientCard): CardRow[] {
  const history = card.has_history_here === false ? "Nie" : card.has_history_here === true ? "Tak" : "—";
  const rows: CardRow[] = [
    { label: "Imię i nazwisko", value: card.name || "—" },
    { label: "Wiek", value: card.age || "—" },
    { label: "Historia w punkcie", value: history },
  ];
  if (card.has_history_here === false) return rows;
  rows.push(
    { label: "Choroby przewlekłe", value: card.chronic_diseases || "—" },
    { label: "Operacje", value: card.operations || "—" },
    { label: "Alergie", value: card.allergies || "—" },
    { label: "Wywiad rodzinny", value: card.family_history || "—" }
  );
  return rows;
}
