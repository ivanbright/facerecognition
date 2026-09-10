#!/usr/bin/env python3
"""
Robust enrollment using the shared ArcFace recognizer.
"""
from working_face_recognition import ArcFaceRecognizer


class RobustArcFaceEnrollment:
    def __init__(self, model_path="models/embedder_arcface.onnx"):
        self.recognizer = ArcFaceRecognizer(model_path=model_path)

    def enroll_person(self, name, auto_save_every=5, save_on_quit=True):
        return self.recognizer.enroll_person(
            name,
            auto_save_every=auto_save_every,
            save_on_quit=save_on_quit,
        )

    def finalize_enrollment(self, name, embeddings):
        return self.recognizer.finalize_enrollment(name, embeddings)


def main():
    print("ROBUST ARCFACE ENROLLMENT")
    print("=" * 40)

    try:
        enroller = RobustArcFaceEnrollment()
        name = input("Enter person name to enroll: ").strip()
        if not name:
            print("No name provided")
            return

        success = enroller.enroll_person(name)
        if success:
            print(f"\nSUCCESS! {name} is now enrolled!")
            print("Next: run recognition mode to test tracking")
        else:
            print(f"\nEnrollment failed for {name}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
