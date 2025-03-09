import React from "react";
import { motion } from "framer-motion";
import "./FutureEnhancements.css";

const FutureEnhancements = () => {
  const enhancements = [
    {
      title: "Data Security & Confidentiality",
      content: "End-to-end encryption for secure data transmission.",
    },
    {
      title: "Voice-Based Medical Assistance",
      content: "Voice-enabled AI assistant for hands-free interaction.",
    },
    {
      title: "Multi-Language Support",
      content: "Medical transcription and diagnostics in multiple languages.",
    },
    {
      title: "Real-Time Data Integration",
      content: "Sync with wearables and hospital databases.",
    },
    {
      title: "Electronic Health Record (EHR) Compatibility",
      content: "Seamless EHR system integration.",
    },
    {
      title: "Self-Learning AI Model",
      content: "Continuous learning for improved accuracy.",
    },
  ];

  return (
    <section id="enhancements" className="future-enhancements">
      <motion.h2
        className="enhancements-title"
        initial={{ opacity: 0, y: -50 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: "easeOut" }}
      >
        Future Enhancements
      </motion.h2>

      <div className="enhancements-list">
        {enhancements.map((item, index) => (
          <motion.div
            key={index}
            className="enhancement-item"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: index * 0.2 }}
            whileHover={{ scale: 1.1 }}
          >
            <h3 className="enhancement-title">{item.title}</h3>
            <p className="enhancement-content">{item.content}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
};

export default FutureEnhancements;
