"""Visit the Streamlit app in a headless browser; click the wake-up button if it is asleep."""
from playwright.sync_api import sync_playwright

URL = "https://busra-flight-delay.streamlit.app/"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(8_000)

    wake = page.get_by_role("button", name="Yes, get this app back up!")
    if wake.count() > 0:
        print("App was asleep - waking it up")
        wake.first.click()
        page.wait_for_timeout(90_000)  # give the app time to boot
    else:
        print("App is awake")
        page.wait_for_timeout(20_000)  # stay connected briefly so the visit counts

    browser.close()
