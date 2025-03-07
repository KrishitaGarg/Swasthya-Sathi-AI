import { useState, useEffect } from "react";
import "./Hero.css";

const Hero = () => {
  const [hovered, setHovered] = useState(false);
  const [displayedText, setDisplayedText] = useState("");
  const fullText =
    " Welcome to Swasthya Sathi AI, a cutting-edge AI-powered medical assistant designed to streamline medical diagnostics, transcription, pathology analysis, and more. This project aims to enhance the efficiency of healthcare professionals by leveraging artificial intelligence for accurate and fast medical assessments.";

  useEffect(() => {
    if (hovered) {
      let index = 0;
      const interval = setInterval(() => {
        setDisplayedText((prev) => prev + fullText[index]);
        index++;
        if (index === fullText.length - 1) {
          clearInterval(interval);
        }
      }, 100);
      return () => clearInterval(interval);
    } else {
      setDisplayedText("");
    }
  }, [hovered]);

  return (
    <section className="hero" id="hero">
      <div className="hero-content">
        <div
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
        >
          {hovered ? (
            <p>{displayedText}</p>
          ) : (
            <>
              <h1>Swasthya Sathi AI</h1>
              <h2 className="sub-header">
                Your Health Companion for Medical Diagnosis & Assistance
              </h2>
            </>
          )}
        </div>
        <button onClick={() => window.location.href = "https://swasthya-sathi-ai.streamlit.app/"} className="try-demo">Try Live Demo</button>
      </div>
    </section>
  );
};

export default Hero;
