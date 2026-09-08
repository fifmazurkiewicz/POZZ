export type SimMode = "doctor_asks" | "patient_asks" | "meta_ask";

export type SimMessage = {
  id: number;
  role: "user" | "assistant" | "system";
  content: string;
};
