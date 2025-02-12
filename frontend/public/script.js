document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const previewImage = document.getElementById('previewImage');

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewImage.style.display = "block";
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('uploadBtn').addEventListener('click', function() {
    const fileInput = document.getElementById('fileInput');
    const resultText = document.getElementById('resultText');

    if (!fileInput.files.length) {
        resultText.innerText = "Please select an image first!";
        return;
    }

    let formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('https://authentiscan-hk43.onrender.com/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        resultText.innerText = `Prediction: ${data.result}`;
    })
    .catch(error => {
        console.error("Error:", error);
        resultText.innerText = "Error detecting image!";
    });
});
