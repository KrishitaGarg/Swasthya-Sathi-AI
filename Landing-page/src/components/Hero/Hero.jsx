import { useState } from "react";
import "./Hero.css";

const Hero = () => {
  const [hovered, setHovered] = useState(false);

  return (
    <section className="hero" id="hero">
      <div className="hero-content">
        <div
          className={`hero-text ${hovered ? "hovered" : ""}`}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
        >
          <h1>Swasthya Sathi AI</h1>
          <h2 className="sub-header">
            Your Health Companion for Medical Diagnosis & Assistance
          </h2>
          {hovered && (
            <p className="description">
              A cutting-edge AI-powered medical assistant designed to streamline
              medical diagnostics, transcription, pathology analysis, and more.
            </p>
          )}
        </div>
        <button
          onClick={() =>
            (window.location.href = "https://swasthya-sathi-ai.streamlit.app/")
          }
          className="try-demo"
        >
          Try Live Demo
        </button>
      </div>
    </section>
  );
};

export default Hero;
