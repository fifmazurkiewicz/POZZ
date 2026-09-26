from app.prompts.simulation import create_examination_prompt, generate_patient_scenario_prompt


def test_scenario_prompt_requires_a_single_consistent_clinical_state():
    content = generate_patient_scenario_prompt()[0]["content"]

    assert "jeden wewnętrzny stan pacjenta" in content
    assert "obrzęków podudzi i braku obrzęków" in content
    assert "Informacje z wywiadu" in content


def test_examination_prompt_separates_history_and_reports_unperformed_tests():
    content = create_examination_prompt(
        patient_scenario="Scenariusz testowy.", chat_history=[], examination="gazometria"
    )[0]["content"]

    assert "Oddziel badanie przedmiotowe od wywiadu" in content
    assert "Niewykonano, ponieważ" in content
    assert "tonów serca jako „głośne”" in content
