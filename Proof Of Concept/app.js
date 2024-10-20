const express = require("express");
const multer = require("multer");
const axios = require("axios");
const path = require("path");
const app = express();

// Set up EJS
app.set("view engine", "ejs");

// Set up public folder for static files (CSS, images)
app.use(express.static("public"));

// Multer configuration for image upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "public/images");
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({ storage });

// Serve the form on the homepage
app.get("/", (req, res) => {
  res.render("index");
});

// Handle form submission and send data to Flask backend
app.post("/submit", upload.single("image"), async (req, res) => {
  const imagePath = req.file.path;
  const paragraph = req.body.paragraph;

  try {
    // Send image and paragraph to Flask backend
    const response = await axios.post("http://127.0.0.1:5000/process", {
      imagePath,
      paragraph,
    });

    // Get structured output, system scope, and similarity from Flask backend
    const { structuredOutput, systemScope, similarity } = response.data;

    res.render("result", {
      structuredOutput,
      systemScope,
      similarity,
    });
  } catch (error) {
    console.error("Error:", error);
    res.render("index", { error: "Something went wrong. Please try again." });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
