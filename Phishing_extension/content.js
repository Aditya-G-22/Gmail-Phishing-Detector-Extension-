console.log("Gmail Phishing Detector loaded");

function createWarningBanner(level, probability) {
    const banner = document.createElement("div");

    let bgColor, border, text;

    if (level === "high_risk") {
        bgColor = "#fdecea";
        border = "#d93025";
        text = `🚨 High risk phishing detected (${(probability * 100).toFixed(1)}%)`;
    } else {
        bgColor = "#fff4ce";
        border = "#f9ab00";
        text = `⚠️ Suspicious email detected (${(probability * 100).toFixed(1)}%)`;
    }

    banner.id = "phishing-warning-banner";
    banner.style.background = bgColor;
    banner.style.border = `1px solid ${border}`;
    banner.style.padding = "12px";
    banner.style.marginBottom = "8px";
    banner.style.borderRadius = "6px";
    banner.style.fontSize = "14px";
    banner.style.fontWeight = "500";
    banner.style.color = "#202124";

    banner.textContent = text;
    return banner;
}

function injectBanner(result) {
    const emailBody = document.querySelector("div.a3s");
    if (!emailBody) return;

    // Remove old banner if present
    const existing = document.getElementById("phishing-warning-banner");
    if (existing) existing.remove();

    if (result.label === "legitimate") return;

    const level = result.final_probability >= 0.8
    ? "high_risk"
    : "suspicious";

    const banner = createWarningBanner(level, result.final_probability);

    emailBody.parentNode.insertBefore(banner, emailBody);
}


let lastEmailText = "";

function extractEmailText() {
    const emailBody = document.querySelector("div.a3s");
    if (!emailBody) return null;
    return emailBody.innerText.trim();
}

function extractUrls(text) {
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    return text.match(urlRegex) || [];
}


const observer = new MutationObserver(() => {
    const emailText = extractEmailText();

    if (emailText && emailText !== lastEmailText) {
        lastEmailText = emailText;
        console.log("New email detected:");
        console.log(emailText.slice(0, 200));
        sendToBackend(emailText)
    }
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

function sendToBackend(emailText) {
    chrome.runtime.sendMessage(
        {
            type: "PREDICT_EMAIL",
            text: emailText,
            urls: extractUrls(emailText)
        },
        (response) => {
            if (!response || !response.success) {
                console.error("Backend error:", response?.error);
                return;
            }

            console.log("Prediction result:", response.data);
            injectBanner(response.data)
        }
    );
}


