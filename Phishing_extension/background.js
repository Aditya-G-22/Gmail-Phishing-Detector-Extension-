chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "PREDICT_EMAIL") {

        fetch("http://127.0.0.1:8001/predict/email", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: message.text })
        })
        .then(response => response.json())
        .then(data => {
            sendResponse({ success: true, data });
        })
        .catch(error => {
            sendResponse({ success: false, error: error.toString() });
        });

        // REQUIRED: tells Chrome response is async
        return true;
    }
    console.log("Calling backend with:", message.text.slice(0, 50));
});
