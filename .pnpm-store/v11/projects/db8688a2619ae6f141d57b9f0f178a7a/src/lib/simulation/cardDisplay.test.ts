import { describe, expect, it } from "vitest";
import { cardRowsForDisplay } from "./cardDisplay";

describe("cardRowsForDisplay", () => {
  it("shows only identity for first-time patients", () => {
    const rows = cardRowsForDisplay({
      name: "Piotr Wiśniewski",
      age: "29 lat",
      has_history_here: false,
      chronic_diseases: "should hide",
      operations: "should hide",
      allergies: "should hide",
      family_history: "should hide",
    });
    expect(rows.map((r) => r.label)).toEqual(["Imię i nazwisko", "Wiek", "Historia w punkcie"]);
    expect(rows[2].value).toBe("Nie");
  });

  it("shows the full card for returning patients", () => {
    const rows = cardRowsForDisplay({
      name: "Anna Nowak",
      age: "42 lata",
      has_history_here: true,
      chronic_diseases: "nadciśnienie",
      operations: "brak",
      allergies: "penicylina",
      family_history: "ojciec zawał",
    });
    expect(rows).toHaveLength(7);
    expect(rows[3].value).toBe("nadciśnienie");
  });
});
