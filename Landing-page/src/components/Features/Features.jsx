import { useState, useEffect } from "react";
import "./Features.css";
import { motion } from "framer-motion";

const featuresList = [
  {
    title: "AI-Assisted Imaging",
    description:
      "Leverages AI to analyze medical images with precision and speed.",
  },
  {
    title: "Smart Medical Transcriber",
    description: "Automatically converts medical dictations into text format.",
  },
  {
    title: "Lab Report Analyzer",
    description: "Interprets pathology reports and highlights abnormalities.",
  },
  {
    title: "Medical Coding",
    description: "Suggests ICD medical codes for medical reports.",
  },
  {
    title: "Insurance Risk Evaluator",
    description: "Assesses patient data to predict insurance risk.",
  },
  {
    title: "Treatment & Diet Planner",
    description: "Generates personalized treatment and diet plans.",
  },
  {
    title: "Disease Predictor",
    description: "Predicts diseases based on symptoms chosen by user.",
  },
  {
    title: "AI Chatbot",
    description: "Chatbot that predicts diseases based on symptoms.",
  },
];

const Features = () => {
  const [hoveredIndex, setHoveredIndex] = useState(null);

  useEffect(() => {
    const featuresSection = document.getElementById("features");
    const handleScroll = () => {
      const sectionPos = featuresSection.getBoundingClientRect().top;
      const screenHeight = window.innerHeight;

      if (sectionPos < screenHeight * 0.75) {
        featuresSection.classList.add("fadeInAnimated");
      } else {
        featuresSection.classList.remove("fadeInAnimated");
      }
    };

    window.addEventListener("scroll", handleScroll);
    handleScroll();

    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <section id="features" className="features">
      <motion.h2
        className="features-title"
        initial={{ opacity: 0, y: -50 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: "easeOut" }}
      >
        Our Features
      </motion.h2>

      <div
        className="features-grid"
      >
        {featuresList.map((feature, index) => (
          <motion.div
            key={index}
            className={`feature-card ${
              hoveredIndex === index ? "flipped" : ""
            }`}
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: index * 0.2 }}
            whileHover={{ scale: 1.02 }}
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            <div className="card-inner">
              <div className="card-front">{feature.title}</div>
              <div className="card-back">
                <p>{feature.description}</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
};

export default Features;
