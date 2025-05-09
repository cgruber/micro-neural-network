package me.cgruber.dl.ocr

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.ProgramResult
import com.github.ajalt.clikt.core.main
import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.parameters.options.convert
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.option
import java.io.File
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist

class Ocr : CliktCommand(name = "ocr") {
  override fun run() {}
}

class Train : CliktCommand() {
  val dir by option("--dir", "-d", help = "Cache directory").convert { File(it) }.default(File("."))
  val modelFile by
    option("--model", "-m", help = "The model persistence file")
      .convert { File(it) }
      .default(File("models/my.model"))

  override fun run() {
    echo("Testing Train")
    if (true) throw ProgramResult(0)
    val (training, test) = fashionMnist(dir)
    val model =
      Sequential.of(
        // comment to keep vertical style
        Input(28, 28, 1),
        Flatten(),
        Dense(100),
        Dense(50),
        Dense(10),
      )
    TensorFlowInferenceModel.load(modelFile).use {
      it.reshape(28, 28, 1)
      val prediction = it.predict(test.getX(0))
      val actualLabel = test.getY(0)
      println(
        "Predicted label is: $prediction. This corresponds to class ${stringLabels[prediction]}."
      )
      println("Actual label is: $actualLabel.")
    }
  }
}

class Run : CliktCommand() {
  override fun run() {
    echo("Testing Run")
    if (true) throw ProgramResult(0)
  }
}

val stringLabels =
  mapOf(
    0 to "T-shirt/top",
    1 to "Trousers",
    2 to "Pullover",
    3 to "Dress",
    4 to "Coat",
    5 to "Sandals",
    6 to "Shirt",
    7 to "Sneakers",
    8 to "Bag",
    9 to "Ankle boots",
  )

fun main(vararg args: String) {
  Ocr().subcommands(Train(), Run()).main(args)
}
