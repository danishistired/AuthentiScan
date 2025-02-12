import React, { useState } from "react";
import axios from "axios";
import "./style.css"; // Import the CSS file

const ImageUpload = () => {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState("");

  const handleFileChange = (event) => {
    setImage(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!image) {
      alert("Please select an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData);
      setResult(response.data.result);
    } catch (error) {
      console.error("Error uploading image:", error);
      setResult("Error processing image");
    }
  };

  return (
    <div className="container">
      <h1>Deepfake Detector</h1>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload & Analyze</button>
      {result && <p className="result">{result}</p>}
    </div>
  );
};

export default ImageUpload;
