from hiver_agent.text import is_unhandleable


def test_v2_catches_promo_brand_and_non_english():
    assert is_unhandleable("Premium gives you unlimited skips. Get 3 months now for just 99p.", "v2")
    assert is_unhandleable("2: Can you DM it to us instead? We'll be waiting /JE", "v2")
    assert is_unhandleable("Waarom worden mijn afspeellijsten steeds opnieuw gedownload als ik Spotify open?", "v2")
    assert not is_unhandleable("where is lemonade", "v2")
    assert not is_unhandleable("can you add Jay-Z please", "v2")
    assert not is_unhandleable("Gracias fam, can we get the Vicente Fernandez lyrics up on the app?", "v2")
